"""Unit tests for shared sessions, timeout selection, retries, and health checks."""

from dataclasses import replace
from unittest.mock import Mock, call

import pytest
import requests

URL = "https://provider.example/v1/chat/completions"
PAYLOAD = {"messages": [{"role": "user", "content": "Hello"}]}


def response(status=200):
    result = requests.Response()
    result.status_code = status
    result.url = URL
    return result


@pytest.fixture
def session(net, monkeypatch):
    result = Mock(spec=requests.Session)
    result.get.return_value = response()
    result.post.return_value = response()
    monkeypatch.setattr(net, "_session", result)
    return result


def send(net, method, **kwargs):
    if method == "post":
        return net.http_post(URL, PAYLOAD, **kwargs)
    return net.http_get(URL, **kwargs)


def test_make_session_configures_both_transports(net, monkeypatch):
    monkeypatch.setattr(net, "NET", replace(net.NET, max_retries=4, backoff_base=0.6))

    with net.make_session() as session:
        adapter = session.get_adapter("https://")
        assert adapter is session.get_adapter("http://")
        assert adapter.max_retries.total == 0
        assert "POST" not in adapter.max_retries.allowed_methods


def test_make_session_without_urllib3_retry_still_returns_usable_session(net, monkeypatch):
    monkeypatch.setattr(net, "Retry", None)

    with net.make_session() as session:
        assert isinstance(session, requests.Session)
        assert session.get_adapter("http://").max_retries.total == 0
        assert session.get_adapter("https://").max_retries.total == 0


def test_get_session_initializes_lazily_and_reuses_the_same_session(net, monkeypatch):
    factory = Mock(return_value=Mock(spec=requests.Session))
    monkeypatch.setattr(net, "make_session", factory)

    factory.assert_not_called()
    assert net.get_session() is factory.return_value
    assert net.get_session() is factory.return_value
    factory.assert_called_once_with()


@pytest.mark.parametrize("method", ["get", "post"])
def test_request_forwards_payload_headers_and_explicit_timeout(net, session, method):
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
    monkeypatch.setattr(net, "NET", replace(net.NET, local_timeout=32.5, cloud_timeout=11.5))

    net.http_post(url, PAYLOAD)

    assert session.post.call_args.kwargs["timeout"] == getattr(net.NET, timeout_field)


def test_get_uses_configured_health_timeout(net, session, monkeypatch):
    monkeypatch.setattr(net, "NET", replace(net.NET, health_timeout=0.125))

    net.http_get(URL)

    session.get.assert_called_once_with(URL, timeout=0.125, headers=None)


@pytest.mark.parametrize("method", ["get", "post"])
def test_explicit_zero_timeout_is_not_replaced_by_a_default(net, session, method):
    send(net, method, timeout=0)

    assert getattr(session, method).call_args.kwargs["timeout"] == 0


@pytest.mark.parametrize(
    ("method", "error_type"),
    [("get", requests.ConnectionError), ("get", requests.Timeout), ("post", requests.ConnectTimeout)],
)
def test_transient_failures_retry_with_exponential_backoff(net, session, monkeypatch, method, error_type):
    monkeypatch.setattr(net, "NET", replace(net.NET, max_retries=4, backoff_base=0.25))
    success = response()
    transport = getattr(session, method)
    transport.side_effect = [error_type("first"), error_type("second"), error_type("third"), success]

    assert send(net, method) is success

    assert transport.call_count == 4
    assert transport.call_args_list == [transport.call_args] * 4
    assert net.time.sleep.call_args_list == [call(0.25), call(0.5), call(1.0)]


@pytest.mark.parametrize(
    ("method", "error_type"),
    [("get", requests.ConnectionError), ("get", requests.Timeout), ("post", requests.ConnectTimeout)],
)
def test_exhaustion_reports_the_last_failure_and_does_not_sleep_afterwards(
    net, session, method, error_type,
):
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


@pytest.mark.parametrize("error_type", [requests.ConnectionError, requests.ReadTimeout, requests.Timeout])
def test_post_does_not_retry_failures_that_may_follow_transmission(net, session, error_type):
    failure = error_type("response lost after request was sent")
    session.post.side_effect = [failure, response()]

    with pytest.raises(net.NetworkError) as caught:
        net.http_post(URL, PAYLOAD)

    assert caught.value.__cause__ is failure
    assert session.post.call_count == 1
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize("method", ["get", "post"])
def test_single_attempt_budget_never_sleeps(net, session, monkeypatch, method):
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
    failure = response(status)
    getattr(session, method).return_value = failure

    with pytest.raises(requests.HTTPError) as caught:
        send(net, method)

    assert caught.value.response is failure
    assert getattr(session, method).call_count == 1
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
@pytest.mark.parametrize("transport_retry_available", [True, False])
def test_post_does_not_retry_http_errors_after_transmission(
    net, session, monkeypatch, status, transport_retry_available,
):
    if not transport_retry_available:
        monkeypatch.setattr(net, "Retry", None)
    failure = response(status)
    session.post.side_effect = [failure, response()]

    with pytest.raises(requests.HTTPError) as caught:
        net.http_post(URL, PAYLOAD)

    assert caught.value.response is failure
    assert session.post.call_count == 1
    net.time.sleep.assert_not_called()


def test_post_stops_on_http_error_after_safe_connect_timeout_retry(net, session):
    failure = response(503)
    session.post.side_effect = [requests.ConnectTimeout("connect timed out"), failure, response()]

    with pytest.raises(requests.HTTPError) as caught:
        net.http_post(URL, PAYLOAD)

    assert caught.value.response is failure
    assert session.post.call_count == 2
    net.time.sleep.assert_called_once_with(1.0)


def test_post_stops_on_ambiguous_failure_after_safe_connect_timeout_retry(net, session):
    failure = requests.ConnectionError("connection reset after transmission")
    session.post.side_effect = [requests.ConnectTimeout("connect timed out"), failure, response()]

    with pytest.raises(net.NetworkError) as caught:
        net.http_post(URL, PAYLOAD)

    assert caught.value.__cause__ is failure
    assert session.post.call_count == 2
    net.time.sleep.assert_called_once_with(1.0)


@pytest.mark.parametrize("method", ["get", "post"])
def test_http_errors_are_not_retried(net, session, method):
    getattr(session, method).return_value = response(503)

    with pytest.raises(requests.HTTPError):
        send(net, method)

    assert getattr(session, method).call_count == 1
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize("method", ["get", "post"])
def test_unexpected_request_errors_are_not_reclassified_as_network_exhaustion(net, session, method):
    failure = requests.exceptions.InvalidURL("bad URL")
    getattr(session, method).side_effect = failure

    with pytest.raises(requests.exceptions.InvalidURL) as caught:
        send(net, method)

    assert caught.value is failure
    assert getattr(session, method).call_count == 1
    net.time.sleep.assert_not_called()


@pytest.mark.parametrize("status", [200, 204, 301, 400, 404, 499, 500, 503])
def test_is_alive_status_boundary(net, session, status):
    session.get.return_value = response(status)

    assert net.is_alive(URL) is (status < 500)

    session.get.assert_called_once_with(URL, timeout=net.NET.health_timeout)
    net.time.sleep.assert_not_called()


def test_is_alive_forwards_explicit_timeout(net, session):
    assert net.is_alive(URL, timeout=0.25) is True

    session.get.assert_called_once_with(URL, timeout=0.25)


@pytest.mark.parametrize("error_type", [requests.ConnectionError, requests.Timeout, requests.RequestException])
def test_is_alive_converts_request_failure_to_false_without_manual_retries(net, session, error_type):
    session.get.side_effect = error_type("unavailable")

    assert net.is_alive(URL) is False

    session.get.assert_called_once_with(URL, timeout=net.NET.health_timeout)
    net.time.sleep.assert_not_called()
