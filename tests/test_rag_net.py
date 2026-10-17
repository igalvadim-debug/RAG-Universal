"""Unit tests for shared sessions, timeout selection, retries, and health checks."""

from dataclasses import replace
from unittest.mock import Mock, call

import pytest
import requests

URL = "https://provider.example/v1/chat/completions"
PAYLOAD = {"messages": [{"role": "user", "content": "Hello"}]}


def response(status=200):
    """Build a requests.Response stub with the given status code and test URL."""
    result = requests.Response()
    result.status_code = status
    result.url = URL
    return result


@pytest.fixture
def session(net, monkeypatch):
    """Install a mocked requests.Session as rag_net's shared session and return it."""
    result = Mock(spec=requests.Session)
    result.get.return_value = response()
    result.post.return_value = response()
    monkeypatch.setattr(net, "_session", result)
    return result


def send(net, method, **kwargs):
    """Dispatch to rag_net.http_post or http_get for the given method name."""
    if method == "post":
        return net.http_post(URL, PAYLOAD, **kwargs)
    return net.http_get(URL, **kwargs)


def test_make_session_configures_both_transports(net, monkeypatch):
    """make_session should mount the same adapter on http:// and https://, configured from NET."""
    monkeypatch.setattr(net, "NET", replace(net.NET, max_retries=4, backoff_base=0.6))

    with net.make_session() as session:
        adapter = session.get_adapter("https://")
        assert adapter is session.get_adapter("http://")
        assert adapter.max_retries.total == 4
        assert adapter.max_retries.backoff_factor == 0.3
        assert set(adapter.max_retries.status_forcelist) == {429, 500, 502, 503, 504}
        assert adapter.max_retries.allowed_methods == frozenset({"GET", "POST"})
        assert adapter.max_retries.respect_retry_after_header is True


def test_make_session_without_urllib3_retry_still_returns_usable_session(net, monkeypatch):
    """make_session should return a usable Session with transport retries disabled if Retry is unavailable."""
    monkeypatch.setattr(net, "Retry", None)

    with net.make_session() as session:
        assert isinstance(session, requests.Session)
        assert session.get_adapter("http://").max_retries.total == 0
        assert session.get_adapter("https://").max_retries.total == 0


def test_get_session_initializes_lazily_and_reuses_the_same_session(net, monkeypatch):
    """get_session should only call make_session once and cache its result."""
    factory = Mock(return_value=Mock(spec=requests.Session))
    monkeypatch.setattr(net, "make_session", factory)

    factory.assert_not_called()
    assert net.get_session() is factory.return_value
    assert net.get_session() is factory.return_value
    factory.assert_called_once_with()


@pytest.mark.parametrize("method", ["get", "post"])
def test_request_forwards_payload_headers_and_explicit_timeout(net, session, method):
    """http_get/http_post should forward the given headers, timeout, and (for post) JSON payload."""
    headers = {"Authorization": "Bearer synthetic-test-key"}

    result = send(net, method, timeout=1.25, headers=headers)

    transport = getattr(session, method)
    expected = {"headers": headers, "timeout": 1.25}
    if method == "post":
        expected["json"] = PAYLOAD
    transport.assert_called_once_with(URL, **expected)
    assert result is transport.return_value
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize(
    ("url", "timeout_field"),
    [
        ("http://127.0.0.1:5001/v1/chat/completions", "local_timeout"),
        ("http://localhost:8080/v1/chat/completions", "local_timeout"),
        ("https://provider.example/v1/chat/completions", "cloud_timeout"),
        ("http://192.0.2.1:8080/v1/chat/completions", "cloud_timeout"),
    ],
)
def test_post_selects_default_timeout_for_url(net, session, monkeypatch, url, timeout_field):
    """http_post without an explicit timeout should pick local or cloud timeout based on the URL."""
    monkeypatch.setattr(net, "NET", replace(net.NET, local_timeout=32.5, cloud_timeout=11.5))

    net.http_post(url, PAYLOAD)

    assert session.post.call_args.kwargs["timeout"] == getattr(net.NET, timeout_field)


def test_get_uses_configured_health_timeout(net, session, monkeypatch):
    """http_get without an explicit timeout should use NET.health_timeout."""
    monkeypatch.setattr(net, "NET", replace(net.NET, health_timeout=0.125))

    net.http_get(URL)

    session.get.assert_called_once_with(URL, timeout=0.125, headers=None)


@pytest.mark.parametrize("method", ["get", "post"])
def test_explicit_zero_timeout_is_not_replaced_by_a_default(net, session, method):
    """A timeout of 0 should be passed through as-is instead of being treated as unset."""
    send(net, method, timeout=0)

    assert getattr(session, method).call_args.kwargs["timeout"] == 0


@pytest.mark.parametrize("error_type", [requests.ConnectionError])
def test_post_transient_failures_retry_with_exponential_backoff(net, session, monkeypatch, error_type):
    """Connection errors on POST should be retried with exponential backoff up to max_retries."""
    monkeypatch.setattr(net, "NET", replace(net.NET, max_retries=4, backoff_base=0.25))
    success = response()
    transport = session.post
    transport.side_effect = [error_type("first"), error_type("second"), error_type("third"), success]

    assert net.http_post(URL, PAYLOAD) is success

    assert transport.call_count == 4
    assert transport.call_args_list == [transport.call_args] * 4
    assert net.time.sleep.call_args_list == [call(0.25), call(0.5), call(1.0)]


@pytest.mark.parametrize("method", ["get"])
@pytest.mark.parametrize("error_type", [requests.ConnectionError, requests.Timeout])
def test_get_transient_failures_retry_with_exponential_backoff(net, session, monkeypatch, method, error_type):
    """GET connection errors and timeouts should be retried with exponential backoff up to max_retries."""
    monkeypatch.setattr(net, "NET", replace(net.NET, max_retries=4, backoff_base=0.25))
    success = response()
    transport = getattr(session, method)
    transport.side_effect = [error_type("first"), error_type("second"), error_type("third"), success]

    assert send(net, method) is success

    assert transport.call_count == 4
    assert transport.call_args_list == [transport.call_args] * 4
    assert net.time.sleep.call_args_list == [call(0.25), call(0.5), call(1.0)]


def test_post_read_timeout_raises_network_error_without_retry(net, session):
    """A read timeout means the request may have reached the provider: raise NetworkError after one POST."""
    session.post.side_effect = requests.Timeout("read timeout")

    with pytest.raises(net.NetworkError) as caught:
        net.http_post(URL, PAYLOAD)

    assert "таймаут" in str(caught.value)
    assert session.post.call_count == 1
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize("method", ["get"])
@pytest.mark.parametrize("error_type", [requests.ConnectionError])
def test_exhaustion_reports_the_last_failure_and_does_not_sleep_afterwards(
    net, session, method, error_type,
):
    """After connection-error retries are exhausted, NetworkError should report the last failure and stop sleeping."""
    transport = getattr(session, method)
    transport.side_effect = [error_type("first"), error_type("second"), error_type("last failure")]

    with pytest.raises(net.NetworkError) as caught:
        send(net, method)

    assert isinstance(caught.value, RuntimeError)
    assert method.upper() in str(caught.value)
    assert URL in str(caught.value)
    assert "3" in str(caught.value)
    assert "last failure" in str(caught.value)
    assert transport.call_count == 3
    assert net.time.sleep.call_args_list == [call(1.0), call(2.0)]


def test_post_connection_error_exhaustion_uses_one_attempt_budget_without_sleeping(net, session, monkeypatch):
    """With a persistent connection error and max_retries=1, http_post makes one POST and never sleeps."""
    monkeypatch.setattr(net, "NET", replace(net.NET, max_retries=1))
    session.post.side_effect = requests.ConnectionError("connection refused")

    with pytest.raises(net.NetworkError) as caught:
        net.http_post(URL, PAYLOAD)

    assert "connection refused" in str(caught.value)
    assert session.post.call_count == 1
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize("method", ["get", "post"])
def test_single_attempt_budget_never_sleeps(net, session, monkeypatch, method):
    """With max_retries=1, a failing request should raise immediately without sleeping."""
    monkeypatch.setattr(net, "NET", replace(net.NET, max_retries=1))
    transport = getattr(session, method)
    transport.side_effect = requests.Timeout("unavailable")

    with pytest.raises(net.NetworkError):
        send(net, method)

    assert transport.call_count == 1
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_preserve_the_response_and_fail_without_retry(net, session, method, status):
    """4xx client error responses should raise HTTPError immediately, without retrying."""
    failure = response(status)
    getattr(session, method).return_value = failure

    with pytest.raises(requests.HTTPError) as caught:
        send(net, method)

    assert caught.value.response is failure
    assert getattr(session, method).call_count == 1
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_post_does_not_retry_http_errors_after_transmission(net, session, status):
    """An HTTP error response (already transmitted) must raise HTTPError without resending the POST."""
    failure = response(status)
    session.post.return_value = failure

    with pytest.raises(requests.HTTPError) as caught:
        net.http_post(URL, PAYLOAD)

    assert caught.value.response is failure
    assert session.post.call_count == 1
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize("method", ["get", "post"])
def test_http_status_retry_is_left_to_the_configured_transport(net, session, method):
    """A retryable status should raise once, unretried here; transport policy owns any status retries."""
    getattr(session, method).return_value = response(503)

    with pytest.raises(requests.HTTPError):
        send(net, method)

    assert getattr(session, method).call_count == 1
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize("method", ["get", "post"])
def test_unexpected_request_errors_are_not_reclassified_as_network_exhaustion(net, session, method):
    """Non-retryable Requests exceptions should propagate unchanged, not as NetworkError."""
    failure = requests.exceptions.InvalidURL("bad URL")
    getattr(session, method).side_effect = failure

    with pytest.raises(requests.exceptions.InvalidURL) as caught:
        send(net, method)

    assert caught.value is failure
    assert getattr(session, method).call_count == 1
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize("status", [200, 204, 301, 400, 404, 499, 500, 503])
def test_is_alive_status_boundary(net, session, status):
    """Check is_alive's true/false boundary across a range of HTTP status codes."""
    session.get.return_value = response(status)

    assert net.is_alive(URL) is (status < 500)

    session.get.assert_called_once_with(URL, timeout=net.NET.health_timeout)
    net.time.sleep.assert_not_called()


def test_is_alive_forwards_explicit_timeout(net, session):
    """is_alive should forward an explicit timeout to the underlying GET call."""
    assert net.is_alive(URL, timeout=0.25) is True

    session.get.assert_called_once_with(URL, timeout=0.25)


@pytest.mark.parametrize("error_type", [requests.ConnectionError, requests.Timeout, requests.RequestException])
def test_is_alive_converts_request_failure_to_false_without_manual_retries(net, session, error_type):
    """is_alive should return False on a Requests exception, with a single attempt and no retry."""
    session.get.side_effect = error_type("unavailable")

    assert net.is_alive(URL) is False

    session.get.assert_called_once_with(URL, timeout=net.NET.health_timeout)
    net.time.sleep.assert_not_called()
