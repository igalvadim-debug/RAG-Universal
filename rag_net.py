"""
rag_net.py — сетевой слой: переиспользуемый requests.Session + retry/backoff.

Все HTTP-запросы к LLM-бэкендам (локальным и облачным) идут через http_post()/http_get():
  * одно TCP/TLS-соединение переиспользуется (Session),
  * GET повторяется при сетевых сбоях, POST — только при сбоях установки соединения,
  * таймауты берутся из rag_config.NET.
"""

from __future__ import annotations

import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError

try:  # urllib3 доступен всегда вместе с requests, но импорт держим мягким
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    Retry = None  # type: ignore

from rag_config import NET

log = logging.getLogger("rag.net")

RETRY_STATUS = (429, 500, 502, 503, 504)


def make_session() -> requests.Session:
    """Create a session with connection pooling and transport retries disabled.

    Retry budgets are managed by http_post and http_get. http_post retries
    confirmed connection setup failures only; http_get
    retries connection failures and timeouts.
    """
    s = requests.Session()
    if Retry is not None:
        # total=0: все повторы — во внешнем цикле http_post/http_get.
        # allowed_methods явно ограничивает транспортные ретраи безопасными
        # методами (GET/HEAD и т.п.): POST не повторяется после того как
        # запрос мог достичь провайдера (идемпотентность не гарантирована).
        retry = Retry(total=0, allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}))
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
    return s


# Глобальная сессия приложения (ленивая)
_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Return the shared session, creating and caching it on first use."""
    global _session
    if _session is None:
        _session = make_session()
    return _session


class NetworkError(RuntimeError):
    """Сетевой запрос не удался после всех повторов."""


def _is_pre_transmission_error(exc: requests.ConnectionError) -> bool:
    """Recognize connect timeouts and urllib3 connection setup failures."""
    if isinstance(exc, requests.ConnectTimeout):
        return True
    # NewConnectionError (including DNS failures) is a ConnectTimeoutError.
    return bool(exc.args and isinstance(exc.args[0], MaxRetryError)
                and isinstance(exc.args[0].reason, ConnectTimeoutError))


def http_post(url: str, json_payload: dict, headers: dict | None = None,
              timeout: float | None = None) -> requests.Response:
    """Send JSON and return a response after checking its HTTP status.

    timeout is a connect/read timeout in seconds. None uses NET.cloud_timeout
    for URLs containing '://' but neither '127.0.0.1' nor 'localhost'; other
    URLs use NET.local_timeout.

    Retry only confirmed connection setup failures (connect timeouts, DNS
    failures, or refused connections) with exponential backoff for up to
    NET.max_retries session calls in total. Other connection errors may occur
    after transmission and are terminal. Redirects are disabled; 307/308
    responses raise HTTPError without resending the POST. Any HTTP error
    status, including 429/503, also raises HTTPError immediately, since
    providers do not guarantee POST idempotency.

    Raise NetworkError immediately for ambiguous connection errors or other
    timeouts (including read timeouts), after connection setup attempts are
    exhausted, or if no attempts are configured.
    Other Requests errors also propagate.
    """
    if timeout is None:
        timeout = NET.cloud_timeout if "://" in url and "127.0.0.1" not in url and "localhost" not in url else NET.local_timeout
    sess = get_session()
    last_exc: Exception | None = None
    for attempt in range(1, NET.max_retries + 1):
        try:
            resp = sess.post(url, json=json_payload, headers=headers, timeout=timeout, allow_redirects=False)
            if resp.status_code in (307, 308):
                raise requests.HTTPError(
                    f"POST {url} returned HTTP {resp.status_code}; resending POST to a redirect is forbidden",
                    response=resp,
                )
            # Ответ получен — запрос уже передан провайдеру. POST не повторяем:
            # идемпотентность не гарантирована, любой HTTP-ошибочный статус
            # (включая 429/5xx) уходит наружу через raise_for_status().
            resp.raise_for_status()
            return resp
        except requests.ConnectionError as e:
            if not _is_pre_transmission_error(e):
                raise NetworkError(
                    f"POST {url} прерван (запрос мог быть получен провайдером, повтор POST запрещён): {e}"
                ) from e
            last_exc = e
            if attempt < NET.max_retries:
                wait = NET.backoff_base * (2 ** (attempt - 1))
                log.warning("%s при POST %s — попытка %d/%d, жду %.1f c",
                            type(e).__name__, url, attempt, NET.max_retries, wait)
                time.sleep(wait)
        except requests.Timeout as e:
            # ReadTimeout: запрос уже мог достичь провайдера; без гарантии
            # идемпотентности POST не повторяем — сразу наружу.
            raise NetworkError(f"POST {url} прерван по таймауту (запрос мог быть получен провайдером, повтор POST запрещён): {e}") from e
        except requests.HTTPError:
            raise  # HTTP-ошибки и 307/308 — сразу наружу
    raise NetworkError(f"POST {url} не удался после {NET.max_retries} попыток: {last_exc}") from last_exc


def http_get(url: str, timeout: float | None = None,
             headers: dict | None = None) -> requests.Response:
    """Return a GET response after checking its HTTP status.

    timeout is a connect/read timeout in seconds; None uses NET.health_timeout.
    Retry connection failures and timeouts with exponential backoff for up to
    NET.max_retries session calls. HTTP error responses are not retried.

    Raise NetworkError when connection/timeout attempts are exhausted or no
    attempts are configured. HTTPError and other Requests errors propagate.
    """
    if timeout is None:
        timeout = NET.health_timeout
    sess = get_session()
    last_exc: Exception | None = None
    for attempt in range(1, NET.max_retries + 1):
        try:
            resp = sess.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt < NET.max_retries:
                time.sleep(NET.backoff_base * (2 ** (attempt - 1)))
        except requests.HTTPError:
            raise
    raise NetworkError(f"GET {url} не удался после {NET.max_retries} попыток: {last_exc}")


def is_alive(url: str, timeout: float | None = None) -> bool:
    """Return whether a GET's final response has a status below 400.

    Requests exceptions return False. timeout is in seconds; None or zero uses
    NET.health_timeout. This check does not retry.
    """
    try:
        with requests.Session() as sess:  # свежая сессия, adapter без Retry
            r = sess.get(url, timeout=timeout or NET.health_timeout)
            return r.ok  # только 2xx; 404 (и прочее >=400) — unhealthy
    except requests.RequestException:
        return False
