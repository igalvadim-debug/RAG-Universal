"""
rag_net.py — сетевой слой: переиспользуемый requests.Session + retry/backoff.

Все HTTP-запросы к LLM-бэкендам (локальным и облачным) идут через http_post()/http_get():
  * одно TCP/TLS-соединение переиспользуется (Session),
  * сетевые сбои и 429/5xx повторяются с экспоненциальным backoff,
  * таймауты берутся из rag_config.NET.
"""

from __future__ import annotations

import logging
import time

import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 доступен всегда вместе с requests, но импорт держим мягким
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover
    Retry = None  # type: ignore

from rag_config import NET

log = logging.getLogger("rag.net")

RETRY_STATUS = (429, 500, 502, 503, 504)


def make_session() -> requests.Session:
    """Session с пулом соединений и базовыми retry на уровне транспорта."""
    s = requests.Session()
    if Retry is not None:
        retry = Retry(
            total=NET.max_retries,
            backoff_factor=NET.backoff_base / 2.0,  # urllib3 ждёт half-factor'ом
            status_forcelist=RETRY_STATUS,
            allowed_methods=frozenset({"GET", "POST"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=16)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
    return s


# Глобальная сессия приложения (ленивая)
_session: requests.Session | None = None


def get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = make_session()
    return _session


class NetworkError(RuntimeError):
    """Сетевой запрос не удался после всех повторов."""


def http_post(url: str, json_payload: dict, headers: dict | None = None,
              timeout: float | None = None) -> requests.Response:
    """POST с повторными попытками (экспоненциальный backoff).

    Бросает NetworkError, если все попытки исчерпаны.
    """
    if timeout is None:
        timeout = NET.cloud_timeout if "://" in url and "127.0.0.1" not in url and "localhost" not in url else NET.local_timeout
    sess = get_session()
    last_exc: Exception | None = None
    for attempt in range(1, NET.max_retries + 1):
        try:
            resp = sess.post(url, json=json_payload, headers=headers, timeout=timeout)
            # 429/5xx здесь повторяем вручную (для Session без urllib3-Retry)
            if Retry is None and resp.status_code in RETRY_STATUS and attempt < NET.max_retries:
                wait = NET.backoff_base * (2 ** (attempt - 1))
                log.warning("HTTP %s от %s — попытка %d/%d, жду %.1f c",
                            resp.status_code, url, attempt, NET.max_retries, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            if attempt < NET.max_retries:
                wait = NET.backoff_base * (2 ** (attempt - 1))
                log.warning("%s при POST %s — попытка %d/%d, жду %.1f c",
                            type(e).__name__, url, attempt, NET.max_retries, wait)
                time.sleep(wait)
        except requests.HTTPError:
            raise  # 4xx (кроме повторённых 429/5xx) — сразу наружу
    raise NetworkError(f"POST {url} не удался после {NET.max_retries} попыток: {last_exc}")


def http_get(url: str, timeout: float | None = None,
             headers: dict | None = None) -> requests.Response:
    """GET с ретраями (для health-check'ов и т.п.)."""
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
    """Быстрая проверка доступности (1 попытка, без ретраев — для health-опросов)."""
    try:
        r = get_session().get(url, timeout=timeout or NET.health_timeout)
        return r.status_code < 500
    except requests.RequestException:
        return False
