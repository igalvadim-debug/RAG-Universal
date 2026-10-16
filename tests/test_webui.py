"""Database update orchestration and error propagation introduced in PR #3."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import ANY, Mock, call

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def operations():
    return Mock(spec=["update", "clear_cache", "clear_search_cache"])


@pytest.fixture
def webui(monkeypatch, tmp_path, operations):
    """Import the real UI module with its external operations isolated."""
    for name in ("gradio", "rag_core", "update_db", "stats", "project_manager", "tripple", "grid_splitter"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["update_db"].main = operations.update
    sys.modules["rag_core"].clear_cache = operations.clear_cache
    sys.modules["rag_core"].clear_search_cache = operations.clear_search_cache

    # Importing webui creates directories and extends sys.path for workflows.
    # Keep those side effects in pytest's temporary directory and restore path.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    spec = importlib.util.spec_from_file_location("webui", REPO_ROOT / "webui.py")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(tmp_path / "webui.py")
    monkeypatch.setitem(sys.modules, "webui", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("first_messages", "second_messages", "expected"),
    [
        pytest.param(["Scanning", "Indexed"], ["Already indexed"], "Scanning\nIndexed\nAlready indexed", id="both-runs"),
        pytest.param([], [], "", id="no-messages"),
        pytest.param(["Indexed"], [], "Indexed", id="silent-second-run"),
        pytest.param([], ["Indexed"], "Indexed", id="silent-first-run"),
        pytest.param(
            ["  Индексация 📚  ", ""], ["same", "same", "line one\nline two"],
            "  Индексация 📚  \n\nsame\nsame\nline one\nline two", id="verbatim-messages",
        ),
    ],
)
def test_update_runs_twice_with_cleanup_between_runs_and_returns_only_logs(
    webui, operations, first_messages, second_messages, expected,
):
    batches = iter([first_messages, second_messages])

    def update(*, log_callback):
        for message in next(batches):
            log_callback(message)
        return "Updater return values are not log messages"

    operations.update.side_effect = update
    operations.clear_cache.return_value = "Model cache cleared"
    operations.clear_search_cache.return_value = "Search cache cleared"

    assert webui.do_update_db() == expected

    assert operations.mock_calls == [
        call.update(log_callback=ANY),
        call.clear_cache(),
        call.update(log_callback=ANY),
        call.clear_search_cache(),
    ]


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_first_update_failure_still_cleans_up_and_stops_processing(webui, operations, error_type):
    failure = error_type("Indexing interrupted after a partial write")

    def update(*, log_callback):
        log_callback("Partial progress")
        raise failure

    operations.update.side_effect = update

    with pytest.raises(error_type) as caught:
        webui.do_update_db()

    assert caught.value is failure
    assert operations.mock_calls == [call.update(log_callback=ANY), call.clear_cache()]


@pytest.mark.parametrize("first_update_fails", [False, True], ids=["after-success", "after-failure"])
def test_cleanup_failure_propagates_and_prevents_second_update(webui, operations, first_update_fails):
    indexing_failure = RuntimeError("Indexing failed")
    cleanup_failure = ValueError("Cache cleanup failed")
    if first_update_fails:
        operations.update.side_effect = indexing_failure
    operations.clear_cache.side_effect = cleanup_failure

    with pytest.raises(ValueError) as caught:
        webui.do_update_db()

    assert caught.value is cleanup_failure
    assert caught.value.__context__ is (indexing_failure if first_update_fails else None)
    assert operations.mock_calls == [call.update(log_callback=ANY), call.clear_cache()]


def test_second_update_failure_propagates_without_final_search_cleanup(webui, operations):
    failure = RuntimeError("Second indexing pass failed")
    operations.update.side_effect = [None, failure]

    with pytest.raises(RuntimeError) as caught:
        webui.do_update_db()

    assert caught.value is failure
    assert operations.mock_calls == [
        call.update(log_callback=ANY), call.clear_cache(), call.update(log_callback=ANY),
    ]


def test_final_search_cleanup_failure_propagates(webui, operations):
    failure = RuntimeError("Search cache cleanup failed")
    operations.clear_search_cache.side_effect = failure

    with pytest.raises(RuntimeError) as caught:
        webui.do_update_db()

    assert caught.value is failure
    assert operations.mock_calls == [
        call.update(log_callback=ANY), call.clear_cache(), call.update(log_callback=ANY), call.clear_search_cache(),
    ]


@pytest.mark.parametrize("first_call_fails", [False, True], ids=["after-success", "after-failure"])
def test_each_invocation_starts_with_fresh_log_messages(webui, operations, first_call_fails):
    first_failure = RuntimeError("First request failed")

    def first_update(*, log_callback):
        log_callback("Old progress")
        if first_call_fails:
            raise first_failure

    def next_update(*, log_callback):
        log_callback("New progress")

    operations.update.side_effect = first_update
    if first_call_fails:
        with pytest.raises(RuntimeError) as caught:
            webui.do_update_db()
        assert caught.value is first_failure
    else:
        assert webui.do_update_db() == "Old progress\nOld progress"

    operations.reset_mock()
    operations.update.side_effect = next_update

    assert webui.do_update_db() == "New progress\nNew progress"
    assert operations.mock_calls == [
        call.update(log_callback=ANY), call.clear_cache(), call.update(log_callback=ANY), call.clear_search_cache(),
    ]
