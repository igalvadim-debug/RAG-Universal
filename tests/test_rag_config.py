"""Environment parsing and public configuration contracts introduced in PR #1."""

from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest


@pytest.mark.parametrize("value", [None, "", " ", "\t\n", "  custom value \t"])
def test_env_strips_values_and_falls_back_for_blank_values(config, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TEST_RAG_VALUE", raising=False)
    else:
        monkeypatch.setenv("TEST_RAG_VALUE", value)

    expected = "custom value" if value and value.strip() else "fallback"
    assert config._env("TEST_RAG_VALUE", "fallback") == expected


def test_default_paths_are_rooted_at_the_module(config):
    expected = {
        "model_dir": "model", "chroma_dir": "chroma_zaebalo", "bookrag_dir": "bookrag",
        "docs_dir": "docs", "songs_dir": "Songs", "projects_dir": "projects",
        "env_file": ".env", "stylemusic_path": "stylemusic.json", "settings_path": "settings.txt",
        "debug_log": "webui_debug.log", "workflow_dir": "Workflow",
    }
    assert config.BASE_DIR == Path(config.__file__).resolve().parent
    for field, relative_path in expected.items():
        assert getattr(config.PATHS, field) == config.BASE_DIR / relative_path
    assert config.PATHS.comfy_input_dir == Path(r"D:\ComfyUiVid\input")


def test_path_overrides_are_trimmed_and_allow_spaces_and_unicode(config_factory, tmp_path):
    db = tmp_path / "Datenbank Bücher"
    inputs = tmp_path / "Comfy input"
    config = config_factory(RAG_DB=f"  {db}  ", COMFY_INPUT_DIR=f" {inputs}\t")

    assert config.PATHS.chroma_dir == db
    assert config.PATHS.comfy_input_dir == inputs


@pytest.mark.parametrize("blank", ["", " \t"])
def test_blank_environment_overrides_keep_defaults(config_factory, blank):
    config = config_factory(
        RAG_DB=blank, COMFY_INPUT_DIR=blank, KOBOLD_PORT=blank,
        CLOUD_TIMEOUT=blank, RAG_COLLECTION=blank,
    )

    assert config.PATHS.chroma_dir == config.BASE_DIR / "chroma_zaebalo"
    assert config.PATHS.comfy_input_dir == Path(r"D:\ComfyUiVid\input")
    assert config.NET.kobold_port == 5001
    assert config.NET.cloud_timeout == 120.0
    assert config.RAG.collection == "zaebalo"


def test_network_defaults(config):
    assert asdict(config.NET) == {
        "kobold_host": "127.0.0.1", "kobold_port": 5001,
        "llama_host": "127.0.0.1", "llama_port": 8080, "mtp_port": 8510,
        "remote_llama": "192.168.181.93:7860", "webui_port": 7860,
        "health_timeout": 5.0, "cloud_timeout": 120.0, "local_timeout": 600.0,
        "max_retries": 3, "backoff_base": 1.0,
    }


@pytest.mark.parametrize(
    ("variable", "field", "value", "expected"),
    [
        ("KOBOLD_HOST", "kobold_host", " kobold.internal ", "kobold.internal"),
        ("KOBOLD_PORT", "kobold_port", " 5002 ", 5002),
        ("LLAMA_HOST", "llama_host", " llama.internal ", "llama.internal"),
        ("LLAMA_PORT", "llama_port", "8081", 8081),
        ("MTP_PORT", "mtp_port", "8511", 8511),
        ("REMOTE_LLAMA", "remote_llama", " remote.internal:9000 ", "remote.internal:9000"),
        ("WEBUI_PORT", "webui_port", "7861", 7861),
        ("HEALTH_TIMEOUT", "health_timeout", "0.25", 0.25),
        ("CLOUD_TIMEOUT", "cloud_timeout", "45.5", 45.5),
        ("LOCAL_TIMEOUT", "local_timeout", "90.5", 90.5),
        ("HTTP_RETRIES", "max_retries", "1", 1),
        ("HTTP_BACKOFF_BASE", "backoff_base", "0", 0.0),
    ],
)
def test_network_environment_overrides(config_factory, variable, field, value, expected):
    config = config_factory(**{variable: value})

    actual = getattr(config.NET, field)
    assert actual == expected
    assert type(actual) is type(expected)


def test_rag_defaults(config):
    assert asdict(config.RAG) == {
        "top_k": 7, "collection": "zaebalo", "default_max_tokens": 6000,
        "default_temperature": 0.75,
    }


def test_rag_environment_overrides(config_factory):
    config = config_factory(
        RAG_TOP_K=" 1 ", RAG_COLLECTION=" custom collection ",
        RAG_MAX_TOKENS="128", RAG_TEMPERATURE="0.0",
    )

    assert asdict(config.RAG) == {
        "top_k": 1, "collection": "custom collection", "default_max_tokens": 128,
        "default_temperature": 0.0,
    }


@pytest.mark.parametrize(
    "variable",
    ["KOBOLD_PORT", "LLAMA_PORT", "MTP_PORT", "WEBUI_PORT", "HTTP_RETRIES",
     "HEALTH_TIMEOUT", "CLOUD_TIMEOUT", "LOCAL_TIMEOUT", "HTTP_BACKOFF_BASE",
     "RAG_TOP_K", "RAG_MAX_TOKENS", "RAG_TEMPERATURE"],
)
def test_invalid_numeric_environment_fails_explicitly(config_factory, variable):
    with pytest.raises(ValueError):
        config_factory(**{variable: "not-a-number"})


@pytest.mark.parametrize("name", ["PATHS", "NET", "RAG"])
def test_configuration_instances_are_frozen(config, name):
    instance = getattr(config, name)
    field = next(iter(asdict(instance)))

    with pytest.raises(FrozenInstanceError):
        setattr(instance, field, "changed")


def test_configured_urls_and_llama_endpoint_priority(config):
    net = config.NetConfig(
        kobold_host="kobold.internal", kobold_port=5002,
        llama_host="llama.internal", llama_port=8081, mtp_port=8511,
        remote_llama="remote.internal:9000",
    )

    assert net.kobold_url == "http://kobold.internal:5002/v1/chat/completions"
    assert net.kobold_info_url == "http://kobold.internal:5002/api/v1/info"
    assert net.llama_url == "http://llama.internal:8081/v1/chat/completions"
    assert net.llama_endpoints == [
        ("http://llama.internal:8081/health", "http://llama.internal:8081/v1/chat/completions"),
        ("http://llama.internal:8511/health", "http://llama.internal:8511/v1/chat/completions"),
        ("http://remote.internal:9000/health", "http://remote.internal:9000/v1/chat/completions"),
    ]


def test_remote_endpoint_can_be_omitted_and_endpoint_lists_are_independent(config):
    net = config.NetConfig(remote_llama="")
    endpoints = net.llama_endpoints

    assert len(endpoints) == 2
    endpoints.clear()
    assert len(net.llama_endpoints) == 2


def test_provider_order_and_legacy_agnes_alias(config):
    assert list(config.CLOUD_PROVIDERS) == [
        "OpenAI", "Groq", "DeepSeek", "NVIDIA_NIM", "OpenRouter",
        "Google", "Mistral", "Odirouter", "Agnes",
    ]
    for provider in config.CLOUD_PROVIDERS.values():
        assert set(provider) == {"url", "model", "key_var"}
        assert provider["url"].startswith("https://")
        assert provider["url"].endswith("/chat/completions")
        assert provider["model"].strip()
        assert provider["key_var"].isupper()
        assert provider["key_var"].endswith("_API_KEY")
    assert config.CLOUD_PROVIDERS["Agnes"]["key_var"] == "AGNES_API_KEY"
    assert config.KEY_ALIASES["AGNES_API_KEY"] == ["agnes_API_KEY"]
