"""Tests for configuration wiring, shared HTTP transport, and provider routing."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import requests

MESSAGES = [{"role": "user", "content": "Describe a quiet forest"}]
PROVIDERS = ["OpenAI", "Groq", "DeepSeek", "NVIDIA_NIM", "OpenRouter", "Google", "Mistral", "Odirouter", "Agnes"]


def chat_response(content="Generated text"):
    """Build a mocked chat-completions Response that returns the given message content."""
    response = Mock(spec=requests.Response)
    response.status_code = 200
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


def http_error(status):
    """Build an HTTPError carrying a real Response with the given status code."""
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError("request failed", response=response)


@pytest.fixture
def post(core, monkeypatch):
    """Patch core.http_post with a mock transport and configure deterministic NET timeouts."""
    transport = Mock(return_value=chat_response())
    monkeypatch.setattr(core, "http_post", transport)
    monkeypatch.setattr(core, "NET", replace(core.NET, cloud_timeout=17.5, local_timeout=25.5, max_retries=2))
    return transport


@pytest.mark.parametrize(
    "config",
    [{"RAG_DB": "custom-db", "RAG_COLLECTION": "custom-collection", "RAG_TOP_K": "2",
      "KOBOLD_HOST": "kobold.internal", "KOBOLD_PORT": "5002", "LLAMA_HOST": "llama.internal",
      "LLAMA_PORT": "8081", "MTP_PORT": "8511", "REMOTE_LLAMA": "remote.internal:9000",
      "HEALTH_TIMEOUT": "0.5"}],
    indirect=True,
)
def test_core_uses_shared_configuration(core, config):
    """rag_core's module-level constants should mirror rag_config's PATHS/NET/RAG values."""
    assert core.CLOUD_PROVIDERS is config.CLOUD_PROVIDERS
    assert core.BASE_DIR == config.BASE_DIR
    for constant, field in {
        "MODEL_DIR": "model_dir", "CHROMA_DIR": "chroma_dir", "BOOKRAG_DIR": "bookrag_dir",
        "DOCS_DIR": "docs_dir", "STYLEMUSIC_PATH": "stylemusic_path",
    }.items():
        assert getattr(core, constant) == getattr(config.PATHS, field)
    assert core.TOP_K == 2
    assert core.COLLECTION == "custom-collection"
    assert core.KOBOLD_URL == config.NET.kobold_url
    assert core.LLAMA_URL == config.NET.llama_url
    assert core.LLAMA_ENDPOINTS == config.NET.llama_endpoints
    assert core.LLAMA_HEALTH_URLS == [health for health, _ in config.NET.llama_endpoints]
    assert core.LLAMA_HEALTH_TIMEOUT == 0.5


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("value", ["", " \t", " synthetic-test-key "])
def test_has_api_key_recognizes_nonblank_canonical_keys(core, provider, value):
    """has_api_key should be True only when the provider's canonical key is non-blank."""
    key = core.CLOUD_PROVIDERS[provider]["key_var"]

    assert core.has_api_key({key: value}, provider) is bool(value.strip())
    assert core.has_api_key({}, provider) is False


def test_has_api_key_accepts_legacy_agnes_key_but_not_an_unrelated_provider_key(core):
    """has_api_key should accept the legacy Agnes alias but not another provider's key."""
    assert core.has_api_key({"AGNES_API_KEY": " ", "agnes_API_KEY": " legacy-test-key "}, "Agnes") is True
    assert core.has_api_key({"OPENAI_API_KEY": "synthetic-test-key"}, "Agnes") is False


@pytest.mark.parametrize("provider", PROVIDERS)
def test_cloud_request_uses_shared_transport_and_configured_provider(core, post, provider):
    """_try_cloud_provider should call http_post with the provider's URL, model, and Bearer key."""
    cfg = core.CLOUD_PROVIDERS[provider]
    core.ENV_FILE.write_text(f"{cfg['key_var']}=  synthetic-test-key  \n", encoding="utf-8")

    assert core._try_cloud_provider(provider, MESSAGES, 123, 0.2) == ("Generated text", None)

    post.assert_called_once_with(
        cfg["url"],
        {"model": cfg["model"], "messages": MESSAGES, "temperature": 0.2, "max_tokens": 123},
        headers={"Content-Type": "application/json", "Authorization": "Bearer synthetic-test-key"},
        timeout=17.5,
    )


@pytest.mark.parametrize(
    ("env_text", "expected_key"),
    [
        ("agnes_API_KEY= legacy-test-key \n", "legacy-test-key"),
        ("AGNES_API_KEY=canonical-test-key\nagnes_API_KEY=legacy-test-key\n", "canonical-test-key"),
        ("AGNES_API_KEY= \nagnes_API_KEY=legacy-test-key\n", "legacy-test-key"),
    ],
)
def test_agnes_key_alias_and_canonical_precedence(core, post, env_text, expected_key):
    """The canonical AGNES_API_KEY should take precedence over the legacy alias when both are set."""
    core.ENV_FILE.write_text(env_text, encoding="utf-8")

    assert core._try_cloud_provider("Agnes", MESSAGES, 64, 0.0) == ("Generated text", None)

    assert post.call_args.kwargs["headers"]["Authorization"] == f"Bearer {expected_key}"


@pytest.mark.parametrize("key_name", ["AGNES_API_KEY", "agnes_API_KEY"])
def test_automatic_cloud_selection_accepts_both_agnes_key_spellings(core, post, monkeypatch, key_name):
    """Auto cloud fallback should pick the Agnes provider for the canonical key, not the legacy alias."""
    core.ENV_FILE.write_text(f"{key_name}=synthetic-test-key\n", encoding="utf-8")
    monkeypatch.setattr(core, "get_backend", Mock(return_value="none"))

    assert core.call_llm(MESSAGES, provider="auto") == "Generated text"

    post.assert_called_once()
    assert post.call_args.args[0] == core.CLOUD_PROVIDERS["Agnes"]["url"]


@pytest.mark.parametrize(
    ("first", "skip", "expected_order"),
    [
        (None, None, ["OpenAI", "Groq", "Agnes"]),
        ("Agnes", None, ["Agnes", "OpenAI", "Groq"]),
        (None, "Agnes", ["OpenAI", "Groq", "Agnes"]),
        ("Agnes", "Agnes", ["OpenAI", "Groq", "Agnes"]),
        ("Agnes", "OpenAI", ["Agnes", "Groq", "OpenAI"]),
    ],
)
def test_cloud_chain_includes_legacy_key_in_each_priority_position(
    core, monkeypatch, first, skip, expected_order,
):
    core.ENV_FILE.write_text(
        "OPENAI_API_KEY=openai-test-key\nGROQ_API_KEY=groq-test-key\n"
        "AGNES_API_KEY= \nagnes_API_KEY= legacy-test-key \n",
        encoding="utf-8",
    )
    attempt = Mock(side_effect=lambda name, *_: (None, f"[{name}] unavailable"))
    monkeypatch.setattr(core, "_try_cloud_provider", attempt)

    result = core._try_cloud_chain(MESSAGES, 91, 0.4, "test fallback", first=first, skip=skip)

    assert attempt.call_args_list == [call(name, MESSAGES, 91, 0.4) for name in expected_order]
    assert result.startswith("[Ошибка]")
    assert result.splitlines()[1:] == [f"[{name}] unavailable" for name in expected_order]
    assert core._auto_provider is None
    core.time.sleep.assert_not_called()


@pytest.mark.parametrize(("first", "skip"), [(None, None), ("Agnes", None), (None, "Agnes"), ("Agnes", "Agnes")])
@pytest.mark.parametrize("env_text", [None, "AGNES_API_KEY= \t\nagnes_API_KEY= \t\n"])
def test_cloud_chain_does_not_attempt_missing_or_blank_keys(core, monkeypatch, first, skip, env_text):
    if env_text is not None:
        core.ENV_FILE.write_text(env_text, encoding="utf-8")
    attempt = Mock()
    monkeypatch.setattr(core, "_try_cloud_provider", attempt)

    result = core._try_cloud_chain(MESSAGES, 91, 0.4, "no credentials", first=first, skip=skip)

    assert result.startswith("[Ошибка]")
    assert "ни одного ключа" in result
    attempt.assert_not_called()
    assert core._auto_provider is None
    core.time.sleep.assert_not_called()


@pytest.mark.parametrize(
    "env_text",
    [
        "agnes_API_KEY= legacy-test-key \n",
        "AGNES_API_KEY= \nagnes_API_KEY=legacy-test-key\n",
        "AGNES_API_KEY=canonical-test-key\nagnes_API_KEY=legacy-test-key\n",
    ],
)
def test_automatic_cloud_fallback_caches_agnes_and_reuses_it(core, post, monkeypatch, env_text):
    core.ENV_FILE.write_text(
        f"OPENAI_API_KEY=openai-test-key\n{env_text}", encoding="utf-8",
    )
    monkeypatch.setattr(core, "get_backend", Mock(return_value="none"))
    post.side_effect = [core.NetworkError("OpenAI unavailable"), chat_response(), chat_response("Cached answer")]

    assert core.call_llm(MESSAGES, max_tokens=80, temperature=0.1) == "Generated text"
    assert core._auto_provider == "Agnes"
    assert core.call_llm(MESSAGES, max_tokens=80, temperature=0.1) == "Cached answer"

    assert [request.args[0] for request in post.call_args_list] == [
        core.CLOUD_PROVIDERS[name]["url"] for name in ["OpenAI", "Agnes", "Agnes"]
    ]
    assert core._auto_provider == "Agnes"


def test_failed_cached_legacy_provider_is_retried_after_other_candidates(core, post, monkeypatch):
    core.ENV_FILE.write_text(
        "OPENAI_API_KEY=openai-test-key\nagnes_API_KEY=legacy-test-key\n", encoding="utf-8",
    )
    monkeypatch.setattr(core, "get_backend", Mock(return_value="none"))
    monkeypatch.setattr(core, "_auto_provider", "Agnes")
    post.side_effect = [core.NetworkError("Agnes unavailable"), core.NetworkError("OpenAI unavailable"), chat_response()]

    assert core.call_llm(MESSAGES) == "Generated text"

    assert [request.args[0] for request in post.call_args_list] == [
        core.CLOUD_PROVIDERS[name]["url"] for name in ["Agnes", "OpenAI", "Agnes"]
    ]
    assert core._auto_provider == "Agnes"


@pytest.mark.parametrize("env_text", [None, "AGNES_API_KEY= \nagnes_API_KEY= \n"])
def test_missing_or_blank_keys_do_not_send_a_request(core, post, env_text):
    """_try_cloud_provider should fail fast with no HTTP request when no usable key is configured."""
    if env_text is not None:
        core.ENV_FILE.write_text(env_text, encoding="utf-8")

    result, error = core._try_cloud_provider("Agnes", MESSAGES, 64, 0.0)

    assert result is None
    assert "[Agnes]" in error
    assert "API ключ не найден" in error
    post.assert_not_called()


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
def test_cloud_http_errors_report_real_status_even_for_falsey_responses(core, post, status):
    """The reported HTTP status should reflect the real code even though Response is falsey on error."""
    core.ENV_FILE.write_text("OPENAI_API_KEY=synthetic-test-key\n", encoding="utf-8")
    failure = http_error(status)
    assert not failure.response  # Regression: requests.Response is false for HTTP errors.
    post.side_effect = failure

    result, error = core._try_cloud_provider("OpenAI", MESSAGES, 64, 0.2)

    assert result is None
    assert "[OpenAI]" in error
    assert f"HTTP {status}" in error
    assert ("лимит запросов" in error) is (status in {429, 503})
    assert post.call_count == 1


def test_cloud_http_error_without_response_is_reported(core, post):
    """An HTTPError with no attached response should be reported as HTTP status 0."""
    core.ENV_FILE.write_text("OPENAI_API_KEY=synthetic-test-key\n", encoding="utf-8")
    post.side_effect = requests.HTTPError("missing response")

    result, error = core._try_cloud_provider("OpenAI", MESSAGES, 64, 0.2)

    assert result is None
    assert "HTTP 0" in error


def test_cloud_network_exhaustion_reports_configured_attempt_count(core, post):
    """A NetworkError from http_post should be reported with the configured retry count."""
    core.ENV_FILE.write_text("OPENAI_API_KEY=synthetic-test-key\n", encoding="utf-8")
    post.side_effect = core.NetworkError("exhausted")

    result, error = core._try_cloud_provider("OpenAI", MESSAGES, 64, 0.2)

    assert result is None
    assert "[OpenAI]" in error
    assert "повторено 2 раз" in error
    assert post.call_count == 1


@pytest.mark.parametrize("data", [{}, {"choices": [{}]}, {"choices": [{"message": {}}]}])
def test_cloud_missing_response_fields_are_reported_as_format_errors(core, post, data):
    """A chat response missing expected fields should be reported as an unexpected-format error."""
    core.ENV_FILE.write_text("OPENAI_API_KEY=synthetic-test-key\n", encoding="utf-8")
    post.return_value.json.return_value = data

    result, error = core._try_cloud_provider("OpenAI", MESSAGES, 64, 0.2)

    assert result is None
    assert "[OpenAI] неожиданный формат ответа" in error


def test_cloud_invalid_json_is_reported_as_a_format_error(core, post):
    """A response body that fails JSON decoding should be reported as an unexpected-format error."""
    core.ENV_FILE.write_text("OPENAI_API_KEY=synthetic-test-key\n", encoding="utf-8")
    post.return_value.json.side_effect = requests.exceptions.JSONDecodeError("invalid JSON", "<html>", 0)

    result, error = core._try_cloud_provider("OpenAI", MESSAGES, 64, 0.2)

    assert result is None
    assert "[OpenAI] неожиданный формат ответа" in error


def test_unexpected_cloud_failure_is_logged_and_returned(core, post, caplog):
    """An unexpected exception from the transport should be logged with traceback and returned as an error."""
    core.ENV_FILE.write_text("OPENAI_API_KEY=synthetic-test-key\n", encoding="utf-8")
    post.side_effect = RuntimeError("unexpected transport failure")

    with caplog.at_level("ERROR", logger="rag.core"):
        result, error = core._try_cloud_provider("OpenAI", MESSAGES, 64, 0.2)

    assert result is None
    assert "unexpected transport failure" in error
    assert "OpenAI" in caplog.text
    assert caplog.records[-1].exc_info is not None


@pytest.mark.parametrize("alive", [True, False])
def test_kobold_health_check_uses_configured_info_url(core, monkeypatch, alive):
    """kobold_alive should probe the configured KoboldCpp info URL and return is_alive's result."""
    monkeypatch.setattr(core, "NET", replace(core.NET, kobold_host="kobold.internal", kobold_port=5555))
    probe = Mock(return_value=alive)
    monkeypatch.setattr(core, "is_alive", probe)

    assert core.kobold_alive() is alive

    probe.assert_called_once_with("http://kobold.internal:5555/api/v1/info", timeout=2)


@pytest.mark.parametrize("first_alive", [0, 1, 2, None])
@pytest.mark.parametrize("function", ["llama_alive", "get_llama_url", "get_llama_port"])
def test_llama_health_selection_obeys_priority_and_unavailable_fallback(
    core, monkeypatch, first_alive, function,
):
    """llama_alive/get_llama_url/get_llama_port should pick the first live endpoint in priority order."""
    outcomes = [False] * 3
    if first_alive is not None:
        outcomes[first_alive] = True
    probe = Mock(side_effect=outcomes)
    monkeypatch.setattr(core, "is_alive", probe)
    expected = {
        "llama_alive": first_alive is not None,
        "get_llama_url": core.LLAMA_URL if first_alive is None else core.LLAMA_ENDPOINTS[first_alive][1],
        "get_llama_port": None if first_alive is None else [8080, 8510, 7860][first_alive],
    }

    assert getattr(core, function)() == expected[function]

    count = 3 if first_alive is None else first_alive + 1
    assert probe.call_args_list == [call(url) for url in core.LLAMA_HEALTH_URLS[:count]]


@pytest.mark.parametrize("malformed", ["http://server/health", "http://server:invalid/health"])
def test_llama_port_skips_malformed_live_endpoint_and_logs_warning(core, monkeypatch, caplog, malformed):
    """get_llama_port should skip a live but unparsable endpoint, log a warning, and use the next one."""
    monkeypatch.setattr(core, "LLAMA_ENDPOINTS", [(malformed, "unused"), ("http://server:9000/health", "chat")])
    monkeypatch.setattr(core, "is_alive", Mock(return_value=True))

    with caplog.at_level("WARNING", logger="rag.core"):
        assert core.get_llama_port() == 9000

    assert malformed in caplog.text


@pytest.fixture
def local(core, post, monkeypatch):
    """Configure core for local (llama) call_llm tests with the cloud fallback stubbed out."""
    monkeypatch.setattr(core, "get_backend", Mock(return_value="llama"))
    monkeypatch.setattr(core, "get_llama_url", Mock(return_value="http://llama.internal:9000/v1/chat/completions"))
    monkeypatch.setattr(core, "load_settings", Mock(return_value={}))
    monkeypatch.setattr(core, "_try_cloud_chain", Mock(return_value="Cloud fallback"))
    return core


@pytest.mark.parametrize("provider", ["kobold", "llama", "auto"])
def test_local_request_uses_shared_transport_and_configured_timeout(local, post, provider):
    """call_llm for a local backend should post to the right URL with the configured local timeout."""
    assert local.call_llm(MESSAGES, max_tokens=128, temperature=0.25, provider=provider) == "Generated text"

    url = local.KOBOLD_URL if provider == "kobold" else local.get_llama_url.return_value
    post.assert_called_once_with(
        url, {"model": "local", "messages": MESSAGES, "temperature": 0.25, "max_tokens": 128}, timeout=25.5,
    )
    local._try_cloud_chain.assert_not_called()


def test_shared_transport_preserves_local_sampling_and_thinking_settings(local, post):
    """call_llm should forward sampling settings and disable thinking when settings.txt requests it."""
    sampling = {"top_p": 0.8, "top_k": 20, "min_p": 0.1, "repeat_penalty": 1.1,
                "presence_penalty": 0.2, "frequency_penalty": 0.3, "seed": 0}
    local.load_settings.return_value = {**sampling, "disable_thinking": True, "ctx_size": 4096}

    local.call_llm(MESSAGES, provider="llama")

    assert post.call_args.args[1] == {
        "model": "local", "messages": MESSAGES, "temperature": 0.75, "max_tokens": 6000,
        **sampling, "think": False,
    }


@pytest.mark.parametrize("content", [None, "", " \t\n"])
def test_empty_local_response_retries_once_without_thinking_and_doubles_budget(local, post, content):
    """An empty/blank local response should trigger one retry with thinking off and double max_tokens."""
    post.side_effect = [chat_response(content), chat_response("Recovered text")]

    assert local.call_llm(MESSAGES, max_tokens=64, temperature=0.0, provider="llama") == "Recovered text"

    first, second = post.call_args_list
    assert first.args[1] == {"model": "local", "messages": MESSAGES, "temperature": 0.0, "max_tokens": 64}
    assert second.args[1] == {**first.args[1], "think": False, "max_tokens": 128}
    assert first.kwargs == second.kwargs == {"timeout": 25.5}
    assert first.args[0] == second.args[0] == local.get_llama_url.return_value
    local._try_cloud_chain.assert_not_called()


@pytest.mark.parametrize("provider", ["kobold", "llama", "auto"])
@pytest.mark.parametrize("kind", ["http", "network", "unexpected"])
def test_local_errors_fall_back_only_for_automatic_provider(local, post, provider, kind):
    """Local request failures should fall back to the cloud chain only when provider is 'auto'."""
    failure = {"http": http_error(503), "network": local.NetworkError("exhausted"),
               "unexpected": ValueError("invalid response")}[kind]
    post.side_effect = failure

    result = local.call_llm(MESSAGES, max_tokens=77, temperature=0.3, provider=provider)

    if provider == "auto":
        assert result == "Cloud fallback"
        args, kwargs = local._try_cloud_chain.call_args
        assert args == (MESSAGES, 77, 0.3)
        assert local.get_llama_url.return_value in kwargs["reason"] or kind == "unexpected"
        if kind == "network":
            assert "повторено 2 раз" in kwargs["reason"]
    else:
        assert result.startswith("[Ошибка]")
        if kind == "http":
            assert result == "[Ошибка] HTTP 503"
        elif kind == "network":
            assert "недоступен" in result and "таймаут" in result
        else:
            assert "invalid response" in result
        local._try_cloud_chain.assert_not_called()
    assert post.call_count == 1


def test_retry_failure_after_empty_local_response_also_falls_back(local, post):
    """If the empty-response retry also fails, auto mode should still fall back to the cloud chain."""
    post.side_effect = [chat_response(""), local.NetworkError("retry failed")]

    assert local.call_llm(MESSAGES, provider="auto") == "Cloud fallback"

    assert post.call_count == 2
    local._try_cloud_chain.assert_called_once()


def test_explicit_local_http_error_without_response_does_not_crash(local, post):
    """An HTTPError with no response should not crash call_llm for an explicit local provider."""
    post.side_effect = requests.HTTPError("no response")

    assert local.call_llm(MESSAGES, provider="llama").startswith("[Ошибка] HTTP")

    local._try_cloud_chain.assert_not_called()


@pytest.fixture
def shared_session(core, net, monkeypatch):
    """Keep the real HTTP helper so request counts include its retries."""
    session = Mock(spec=requests.Session)
    session.post.return_value = chat_response()
    monkeypatch.setattr(net, "_session", session)
    settings = replace(core.NET, cloud_timeout=17.5, local_timeout=25.5, max_retries=2, backoff_base=0.125)
    monkeypatch.setattr(core, "NET", settings)
    monkeypatch.setattr(net, "NET", settings)
    monkeypatch.setattr(core, "get_llama_url", Mock(return_value="http://llama.internal:9000/v1/chat/completions"))
    monkeypatch.setattr(core, "load_settings", Mock(return_value={}))
    core.ENV_FILE.write_text("OPENAI_API_KEY=synthetic-test-key\n", encoding="utf-8")
    # A request through the old session would duplicate the shared helper's POST.
    legacy_post = Mock(side_effect=AssertionError("LLM requests must use the shared HTTP helper"))
    monkeypatch.setattr(core._http, "post", legacy_post)
    yield session
    legacy_post.assert_not_called()


@pytest.mark.parametrize("provider", ["OpenAI", "kobold", "llama"])
@pytest.mark.parametrize(
    ("outcome", "attempts"),
    [("success", 1), ("connection", 2), ("connect-timeout", 2), ("rate-limit", 2), ("read-timeout", 1)],
)
def test_completion_uses_only_shared_session_and_its_retry_policy(core, net, shared_session, provider, outcome, attempts):
    if outcome != "success":
        failure = {
            "connection": requests.ConnectionError("connection refused"),
            "connect-timeout": requests.ConnectTimeout("connection timed out"),
            "rate-limit": http_error(429).response,
            "read-timeout": requests.ReadTimeout("response timed out"),
        }[outcome]
        shared_session.post.side_effect = [failure, chat_response()]

    result = core.call_llm(MESSAGES, max_tokens=73, temperature=0.2, provider=provider)

    if outcome == "read-timeout":
        assert result.startswith("[Ошибка]" if provider != "OpenAI" else "[OpenAI]")
        assert ("нет соединения" if provider == "OpenAI" else "таймаут") in result
    else:
        assert result == "Generated text"
    if provider == "OpenAI":
        cfg = core.CLOUD_PROVIDERS[provider]
        url, model, timeout = cfg["url"], cfg["model"], 17.5
        headers = {"Content-Type": "application/json", "Authorization": "Bearer synthetic-test-key"}
    else:
        url = core.KOBOLD_URL if provider == "kobold" else core.get_llama_url.return_value
        model, timeout, headers = "local", 25.5, None
    expected_request = call(
        url, json={"model": model, "messages": MESSAGES, "temperature": 0.2, "max_tokens": 73},
        headers=headers, timeout=timeout,
    )
    assert shared_session.post.call_args_list == [expected_request] * attempts
    assert net.time.sleep.call_args_list == ([call(0.125)] if attempts == 2 else [])


@pytest.mark.parametrize("provider", ["OpenAI", "kobold", "llama"])
@pytest.mark.parametrize(("status", "attempts"), [(401, 1), (429, 2), (503, 2)])
def test_completion_preserves_http_errors_after_shared_session_attempts(core, net, shared_session, provider, status, attempts):
    shared_session.post.return_value = http_error(status).response

    result = core.call_llm(MESSAGES, provider=provider)

    assert f"HTTP {status}" in result
    assert shared_session.post.call_count == attempts
    assert net.time.sleep.call_args_list == ([call(0.125)] if attempts == 2 else [])


@pytest.mark.parametrize("provider", ["OpenAI", "kobold", "llama"])
def test_completion_does_not_exceed_shared_connection_retry_budget(core, net, shared_session, provider):
    shared_session.post.side_effect = requests.ConnectionError("connection refused")

    result = core.call_llm(MESSAGES, provider=provider)

    assert ("нет соединения" if provider == "OpenAI" else "таймаут") in result
    assert shared_session.post.call_count == 2
    net.time.sleep.assert_called_once_with(0.125)


@pytest.mark.parametrize("provider", ["kobold", "llama"])
@pytest.mark.parametrize("retry_content", ["Recovered text", "", None])
def test_empty_local_response_uses_shared_session_for_one_thinking_retry(core, shared_session, provider, retry_content):
    core.load_settings.return_value = {"top_p": 0.8, "seed": 0}
    shared_session.post.side_effect = [chat_response(" \t\n"), chat_response(retry_content)]

    assert core.call_llm(MESSAGES, max_tokens=73, temperature=0.0, provider=provider) == retry_content

    url = core.KOBOLD_URL if provider == "kobold" else core.get_llama_url.return_value
    payload = {"model": "local", "messages": MESSAGES, "temperature": 0.0, "max_tokens": 73, "top_p": 0.8, "seed": 0}
    assert shared_session.post.call_args_list == [
        call(url, json=payload, headers=None, timeout=25.5),
        call(url, json={**payload, "think": False, "max_tokens": 146}, headers=None, timeout=25.5),
    ]


@pytest.mark.parametrize("provider", ["kobold", "llama", "auto"])
def test_read_timeout_during_thinking_retry_only_falls_back_in_auto_mode(core, net, shared_session, monkeypatch, provider):
    monkeypatch.setattr(core, "get_backend", Mock(return_value="llama"))
    fallback = Mock(return_value="Cloud fallback")
    monkeypatch.setattr(core, "_try_cloud_chain", fallback)
    shared_session.post.side_effect = [chat_response(""), requests.ReadTimeout("retry response timed out")]

    result = core.call_llm(MESSAGES, max_tokens=73, temperature=0.2, provider=provider)

    if provider == "auto":
        assert result == "Cloud fallback"
        fallback.assert_called_once_with(
            MESSAGES, 73, 0.2,
            reason=f"Локальный сервер {core.get_llama_url.return_value} недоступен (повторено 2 раз)",
        )
    else:
        assert result == "[Ошибка] Сервер недоступен или таймаут запроса."
        fallback.assert_not_called()
    assert shared_session.post.call_count == 2
    assert shared_session.post.call_args.kwargs["json"]["think"] is False
    assert shared_session.post.call_args.kwargs["json"]["max_tokens"] == 146
    net.time.sleep.assert_not_called()


def test_debug_log_writes_utf8_and_always_prints(core, monkeypatch, capsys):
    """_debug_log should append UTF-8 timestamped lines to DEBUG_LOG and also print them."""
    monkeypatch.setattr(core.time, "strftime", Mock(return_value="2026-10-16 12:00:00"))

    core._debug_log("Диагностика ✓")
    core._debug_log("Second line")

    expected = "[2026-10-16 12:00:00] Диагностика ✓\n[2026-10-16 12:00:00] Second line\n"
    assert core.DEBUG_LOG.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().out == expected


def test_debug_log_failure_warns_and_still_prints(core, monkeypatch, tmp_path, caplog, capsys):
    """If writing DEBUG_LOG fails, _debug_log should log a warning but still print the message."""
    monkeypatch.setattr(core, "DEBUG_LOG", tmp_path)  # Opening a directory as a file raises OSError.

    with caplog.at_level("WARNING", logger="rag.core"):
        core._debug_log("Visible diagnostic")

    assert "Visible diagnostic" in capsys.readouterr().out
    assert str(tmp_path) in caplog.text


def test_list_collections_logs_unavailable_database_and_continues(core, monkeypatch, tmp_path, caplog):
    """If one database is unreachable, list_collections should log a warning and still list the other."""
    primary, books = tmp_path / "primary", tmp_path / "books"
    primary.mkdir()
    books.mkdir()
    monkeypatch.setattr(core, "CHROMA_DIR", primary)
    monkeypatch.setattr(core, "BOOKRAG_DIR", books)
    available = Mock()
    available.list_collections.return_value = [SimpleNamespace(name="chapters")]
    core.chromadb.PersistentClient.side_effect = [RuntimeError("database locked"), available]

    with caplog.at_level("WARNING", logger="rag.core"):
        assert core.list_collections() == ["bookrag:chapters"]

    assert "database locked" in caplog.text
    assert str(primary) in caplog.text
    assert core.chromadb.PersistentClient.call_args_list == [call(path=str(primary)), call(path=str(books))]


def test_list_collections_ignores_nonexistent_database_directories(core, monkeypatch, tmp_path):
    """list_collections should return an empty list and skip Chroma entirely when both DBs are missing."""
    monkeypatch.setattr(core, "CHROMA_DIR", tmp_path / "missing-primary")
    monkeypatch.setattr(core, "BOOKRAG_DIR", tmp_path / "missing-books")

    assert core.list_collections() == []

    core.chromadb.PersistentClient.assert_not_called()


def test_get_collection_by_name_caches_client_and_collection(core, monkeypatch, tmp_path):
    """Repeated lookups of the same collection must not reopen the Chroma database."""
    primary = tmp_path / "primary"
    monkeypatch.setattr(core, "CHROMA_DIR", primary)
    client = Mock()
    coll = Mock()
    client.get_or_create_collection.return_value = coll
    core.chromadb.PersistentClient = Mock(return_value=client)

    assert core.get_collection_by_name("docs") is coll
    assert core.get_collection_by_name("zaebalo:docs") is coll
    assert core.get_collection_by_name("docs") is coll

    core.chromadb.PersistentClient.assert_called_once_with(path=str(primary))
    client.get_or_create_collection.assert_called_once_with("docs")
    assert core._collection is coll


def test_get_collection_by_name_keeps_separate_cache_per_database(core, monkeypatch, tmp_path):
    """Prefixed names resolve to their own database; switching back reuses cached clients."""
    primary, books = tmp_path / "primary", tmp_path / "books"
    monkeypatch.setattr(core, "CHROMA_DIR", primary)
    monkeypatch.setattr(core, "BOOKRAG_DIR", books)
    main_client, book_client = Mock(), Mock()
    main_coll, book_coll = Mock(), Mock()
    main_client.get_or_create_collection.return_value = main_coll
    book_client.get_or_create_collection.return_value = book_coll
    core.chromadb.PersistentClient = Mock(side_effect=[main_client, book_client])

    assert core.get_collection_by_name("zaebalo:notes") is main_coll
    assert core.get_collection_by_name("bookrag:chapters") is book_coll
    # Повторный запрос к основной базе — клиент и коллекция берутся из кэша.
    assert core.get_collection_by_name("notes") is main_coll

    assert core.chromadb.PersistentClient.call_args_list == [
        call(path=str(primary)),
        call(path=str(books)),
    ]
    main_client.get_or_create_collection.assert_called_once_with("notes")
    book_client.get_or_create_collection.assert_called_once_with("chapters")


def test_get_collection_reuses_cached_chroma_client(core, monkeypatch, tmp_path):
    """get_collection() and get_collection_by_name() share one client per database path."""
    primary = tmp_path / "primary"
    monkeypatch.setattr(core, "CHROMA_DIR", primary)
    client = Mock()
    coll = Mock()
    client.get_or_create_collection.return_value = coll
    core.chromadb.PersistentClient = Mock(return_value=client)

    assert core.get_collection() is coll
    assert core.get_collection_by_name(core.COLLECTION) is coll

    core.chromadb.PersistentClient.assert_called_once_with(path=str(primary))
