"""Tests for PR #1's configuration wiring and shared-network call sites."""

from dataclasses import replace
import os
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import requests

MESSAGES = [{"role": "user", "content": "Describe a quiet forest"}]
PROVIDERS = ["OpenAI", "Groq", "DeepSeek", "NVIDIA_NIM", "OpenRouter", "Google", "Mistral", "Odirouter", "Agnes"]


def chat_response(content="Generated text"):
    response = Mock(spec=requests.Response)
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError("request failed", response=response)


@pytest.fixture
def post(core, monkeypatch):
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
    key = core.CLOUD_PROVIDERS[provider]["key_var"]

    assert core.has_api_key({key: value}, provider) is bool(value.strip())
    assert core.has_api_key({}, provider) is False


def test_has_api_key_accepts_legacy_agnes_key_but_not_an_unrelated_provider_key(core):
    assert core.has_api_key({"AGNES_API_KEY": " ", "agnes_API_KEY": " legacy-test-key "}, "Agnes") is True
    assert core.has_api_key({"OPENAI_API_KEY": "synthetic-test-key"}, "Agnes") is False


@pytest.mark.parametrize("provider", PROVIDERS)
def test_cloud_request_uses_shared_transport_and_configured_provider(core, post, provider):
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
    core.ENV_FILE.write_text(env_text, encoding="utf-8")

    assert core._try_cloud_provider("Agnes", MESSAGES, 64, 0.0) == ("Generated text", None)

    assert post.call_args.kwargs["headers"]["Authorization"] == f"Bearer {expected_key}"


@pytest.mark.parametrize(
    "key_name",
    [
        "AGNES_API_KEY",
        pytest.param(
            "agnes_API_KEY",
            marks=pytest.mark.xfail(
                strict=True,
                reason="PR #1: _try_cloud_chain filters canonical keys only and skips the legacy Agnes alias",
            ),
        ),
    ],
)
def test_automatic_cloud_selection_accepts_both_agnes_key_spellings(core, post, monkeypatch, key_name):
    core.ENV_FILE.write_text(f"{key_name}=synthetic-test-key\n", encoding="utf-8")
    monkeypatch.setattr(core, "get_backend", Mock(return_value="none"))

    assert core.call_llm(MESSAGES, provider="auto") == "Generated text"

    post.assert_called_once()
    assert post.call_args.args[0] == core.CLOUD_PROVIDERS["Agnes"]["url"]


@pytest.mark.parametrize("env_text", [None, "AGNES_API_KEY= \nagnes_API_KEY= \n"])
def test_missing_or_blank_keys_do_not_send_a_request(core, post, env_text):
    if env_text is not None:
        core.ENV_FILE.write_text(env_text, encoding="utf-8")

    result, error = core._try_cloud_provider("Agnes", MESSAGES, 64, 0.0)

    assert result is None
    assert "[Agnes]" in error
    assert "API ключ не найден" in error
    post.assert_not_called()


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
def test_cloud_http_errors_report_real_status_even_for_falsey_responses(core, post, status):
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
    core.ENV_FILE.write_text("OPENAI_API_KEY=synthetic-test-key\n", encoding="utf-8")
    post.side_effect = requests.HTTPError("missing response")

    result, error = core._try_cloud_provider("OpenAI", MESSAGES, 64, 0.2)

    assert result is None
    assert "HTTP 0" in error


def test_cloud_network_exhaustion_reports_configured_attempt_count(core, post):
    core.ENV_FILE.write_text("OPENAI_API_KEY=synthetic-test-key\n", encoding="utf-8")
    post.side_effect = core.NetworkError("exhausted")

    result, error = core._try_cloud_provider("OpenAI", MESSAGES, 64, 0.2)

    assert result is None
    assert "[OpenAI]" in error
    assert "повторено 2 раз" in error
    assert post.call_count == 1


@pytest.mark.parametrize("data", [{}, {"choices": [{}]}, {"choices": [{"message": {}}]}])
def test_cloud_missing_response_fields_are_reported_as_format_errors(core, post, data):
    core.ENV_FILE.write_text("OPENAI_API_KEY=synthetic-test-key\n", encoding="utf-8")
    post.return_value.json.return_value = data

    result, error = core._try_cloud_provider("OpenAI", MESSAGES, 64, 0.2)

    assert result is None
    assert "[OpenAI] неожиданный формат ответа" in error


def test_cloud_invalid_json_is_reported_as_a_format_error(core, post):
    core.ENV_FILE.write_text("OPENAI_API_KEY=synthetic-test-key\n", encoding="utf-8")
    post.return_value.json.side_effect = requests.exceptions.JSONDecodeError("invalid JSON", "<html>", 0)

    result, error = core._try_cloud_provider("OpenAI", MESSAGES, 64, 0.2)

    assert result is None
    assert "[OpenAI] неожиданный формат ответа" in error


def test_unexpected_cloud_failure_is_logged_and_returned(core, post, caplog):
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
    monkeypatch.setattr(core, "LLAMA_ENDPOINTS", [(malformed, "unused"), ("http://server:9000/health", "chat")])
    monkeypatch.setattr(core, "is_alive", Mock(return_value=True))

    with caplog.at_level("WARNING", logger="rag.core"):
        assert core.get_llama_port() == 9000

    assert malformed in caplog.text


@pytest.fixture
def local(core, post, monkeypatch):
    monkeypatch.setattr(core, "get_backend", Mock(return_value="llama"))
    monkeypatch.setattr(core, "get_llama_url", Mock(return_value="http://llama.internal:9000/v1/chat/completions"))
    monkeypatch.setattr(core, "load_settings", Mock(return_value={}))
    monkeypatch.setattr(core, "_try_cloud_chain", Mock(return_value="Cloud fallback"))
    return core


@pytest.mark.parametrize("provider", ["kobold", "llama", "auto"])
def test_local_request_uses_shared_transport_and_configured_timeout(local, post, provider):
    assert local.call_llm(MESSAGES, max_tokens=128, temperature=0.25, provider=provider) == "Generated text"

    url = local.KOBOLD_URL if provider == "kobold" else local.get_llama_url.return_value
    post.assert_called_once_with(
        url, {"model": "local", "messages": MESSAGES, "temperature": 0.25, "max_tokens": 128}, timeout=25.5,
    )
    local._try_cloud_chain.assert_not_called()


def test_shared_transport_preserves_local_sampling_and_thinking_settings(local, post):
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
    post.side_effect = [chat_response(""), local.NetworkError("retry failed")]

    assert local.call_llm(MESSAGES, provider="auto") == "Cloud fallback"

    assert post.call_count == 2
    local._try_cloud_chain.assert_called_once()


def test_explicit_local_http_error_without_response_does_not_crash(local, post):
    post.side_effect = requests.HTTPError("no response")

    assert local.call_llm(MESSAGES, provider="llama").startswith("[Ошибка] HTTP")

    local._try_cloud_chain.assert_not_called()


def test_debug_log_writes_utf8_and_always_prints(core, monkeypatch, capsys):
    monkeypatch.setattr(core.time, "strftime", Mock(return_value="2026-10-16 12:00:00"))

    core._debug_log("Диагностика ✓")
    core._debug_log("Second line")

    expected = "[2026-10-16 12:00:00] Диагностика ✓\n[2026-10-16 12:00:00] Second line\n"
    assert core.DEBUG_LOG.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().out == expected


def test_debug_log_failure_warns_and_still_prints(core, monkeypatch, tmp_path, caplog, capsys):
    monkeypatch.setattr(core, "DEBUG_LOG", tmp_path)  # Opening a directory as a file raises OSError.

    with caplog.at_level("WARNING", logger="rag.core"):
        core._debug_log("Visible diagnostic")

    assert "Visible diagnostic" in capsys.readouterr().out
    assert str(tmp_path) in caplog.text


def test_list_collections_logs_unavailable_database_and_continues(core, monkeypatch, tmp_path, caplog):
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
    monkeypatch.setattr(core, "CHROMA_DIR", tmp_path / "missing-primary")
    monkeypatch.setattr(core, "BOOKRAG_DIR", tmp_path / "missing-books")

    assert core.list_collections() == []

    core.chromadb.PersistentClient.assert_not_called()


def _freeze_mtime(path, monkeypatch):
    """Имитирует ФС с грубым st_mtime: stat() для этого файла всегда
    возвращает исходную метку времени (нулевая точность часов ФС)."""
    frozen = path.stat().st_mtime
    real_stat = type(path).stat

    def patched_stat(self, *args, **kwargs):
        """Freeze only the target file's mtime while preserving other stat fields."""
        # Не трогаем stat() других файлов — иначе сломается раннер pytest.
        if self != path:
            return real_stat(self, *args, **kwargs)
        st = real_stat(self, *args, **kwargs)
        proxy = Mock(wraps=st)
        proxy.st_mtime = frozen
        # size=None → проксирует st_size и остальной протокол os.stat_result
        # (индексирование, dir()), что нужно внутренностям pathlib/pytest.
        del proxy.size
        return proxy

    monkeypatch.setattr(type(path), "stat", patched_stat)


def test_load_env_detects_change_when_mtime_granularity_is_coarse(core, monkeypatch):
    """Регрессия (CodeRabbit): на ФС с грубым st_mtime (Windows/Docker)
    перезапись .env может не изменить метку времени. Кэш по одной mtime
    вечно отдавал бы устаревшие настройки — подпись должна включать размер."""
    core.ENV_FILE.write_text("API_KEY=old\n", encoding="utf-8")
    assert core.load_env() == {"API_KEY": "old"}

    # Имитируем файловую систему с нулевой точностью mtime:
    # любое stat() возвращает неизменную метку времени.
    _freeze_mtime(core.ENV_FILE, monkeypatch)

    core.ENV_FILE.write_text("API_KEY=new-value-longer\n", encoding="utf-8")
    assert core.load_env() == {"API_KEY": "new-value-longer"}


def test_save_env_invalidates_cache_even_if_size_and_mtime_unchanged(core, monkeypatch):
    """save_env() обязан принудительно инвалидировать кэш: если обновление
    меняет значение на строку той же длины при неизменной mtime, подпись
    файла не изменится и без флага load_env() вернул бы старые данные."""
    core.ENV_FILE.write_text("API_KEY=aaaa\n", encoding="utf-8")
    assert core.load_env() == {"API_KEY": "aaaa"}

    _freeze_mtime(core.ENV_FILE, monkeypatch)

    # Обновление через save_env(): та же длина значения, «неизменная» mtime.
    core.save_env({"API_KEY": "bbbb"})
    assert core.load_env() == {"API_KEY": "bbbb"}
    # После принудительной перезагрузки флаг сброшен и кэш снова активен.
    assert core._env_cache_invalidated is False
    assert core.load_env() == {"API_KEY": "bbbb"}
