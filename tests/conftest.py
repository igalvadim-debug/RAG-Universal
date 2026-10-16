"""Isolate configuration, HTTP, and optional ML dependencies for unit tests."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ENV_VARS = (
    "RAG_DB", "COMFY_INPUT_DIR", "KOBOLD_HOST", "KOBOLD_PORT", "LLAMA_HOST",
    "LLAMA_PORT", "MTP_PORT", "REMOTE_LLAMA", "WEBUI_PORT", "HEALTH_TIMEOUT",
    "CLOUD_TIMEOUT", "LOCAL_TIMEOUT", "HTTP_RETRIES", "HTTP_BACKOFF_BASE",
    "RAG_TOP_K", "RAG_COLLECTION", "RAG_MAX_TOKENS", "RAG_TEMPERATURE",
)


def _load_module(name, monkeypatch):
    """Execute the real module and restore any previously imported copy afterward."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def unexpected_request(*args, **kwargs):
        pytest.fail("Unit tests must mock HTTP requests")

    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_request)


@pytest.fixture
def config_factory(monkeypatch):
    def load(**overrides):
        for name in CONFIG_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        for name, value in overrides.items():
            monkeypatch.setenv(name, value)
        return _load_module("rag_config", monkeypatch)

    return load


@pytest.fixture
def config(config_factory, request):
    return config_factory(**getattr(request, "param", {}))


@pytest.fixture
def net(config, monkeypatch):
    module = _load_module("rag_net", monkeypatch)
    monkeypatch.setattr(module.time, "sleep", Mock())
    yield module
    if isinstance(module._session, requests.Session):
        module._session.close()


@pytest.fixture
def core(net, monkeypatch, tmp_path):
    # These optional packages are only needed by embedding/database operations.
    # Import the complete rag_core module without loading models or opening a DB.
    chromadb = ModuleType("chromadb")
    chromadb.PersistentClient = Mock()
    transformers = ModuleType("sentence_transformers")
    transformers.SentenceTransformer = Mock()
    monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    monkeypatch.setitem(sys.modules, "sentence_transformers", transformers)
    module = _load_module("rag_core", monkeypatch)
    monkeypatch.setattr(module, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(module, "SETTINGS_PATH", tmp_path / "settings.txt")
    monkeypatch.setattr(module, "DEBUG_LOG", tmp_path / "debug.log")
    return module
