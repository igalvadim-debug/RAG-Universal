"""Database update orchestration (PR #3) and grid caption controls (PR #9)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import ANY, MagicMock, Mock, call

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


@pytest.mark.parametrize(("model", "expected_model"), [("Qwen3-VL-4B", "qwen"), ("Gemma-4-E2B", "gemma")])
@pytest.mark.parametrize("with_captions", [True, False])
def test_grid_split_forwards_model_and_caption_choices_and_preserves_gallery_order(
    webui, model, expected_model, with_captions, tmp_path,
):
    folder = tmp_path / "grid1"
    files = [folder / "1abc.png", folder / "2xyz.png"]
    split = Mock(return_value=(folder, files))
    webui.grid_splitter.split_image = split
    caption_choice = "с текстом (подписями)" if with_captions else "без текста (только кадр)"

    status, gallery = webui.do_grid_split("storyboard.png", model, caption_choice)

    split.assert_called_once_with(
        "storyboard.png", vl=expected_model, with_captions=with_captions, on_log=ANY,
    )
    assert callable(split.call_args.kwargs["on_log"])
    assert gallery == [str(file) for file in files]
    assert status == f"✅ 2 Bilder → {folder}  (in used.md eingetragen)"


@pytest.mark.parametrize("caption_choice", [None, "", "без", "без текста", "unknown choice"])
def test_grid_split_only_explicit_no_caption_choice_disables_captions(webui, caption_choice, tmp_path):
    """Regression: a prefix match must not turn an unrecognized choice into trimming."""
    webui.grid_splitter.split_image = Mock(return_value=(tmp_path, []))

    status, gallery = webui.do_grid_split("grid.png", "Qwen3-VL-4B", caption_choice)

    assert webui.grid_splitter.split_image.call_args.kwargs["with_captions"] is True
    assert status.startswith("✅ 0 Bilder")
    assert gallery == []


@pytest.mark.parametrize("image_path", [None, ""])
def test_grid_split_missing_upload_does_not_call_splitter(webui, image_path):
    webui.grid_splitter.split_image = Mock()

    assert webui.do_grid_split(image_path, "Qwen3-VL-4B", "без текста (только кадр)") == (
        "❌ Bitte zuerst ein Gitter-Bild hochladen.", [],
    )

    webui.grid_splitter.split_image.assert_not_called()


@pytest.mark.parametrize("error_type", [RuntimeError, OSError, ValueError])
@pytest.mark.parametrize("message_count", [0, 12, 15])
def test_grid_split_failure_returns_empty_gallery_and_last_twelve_log_messages(
    webui, error_type, message_count,
):
    messages = [f"Сообщение {index}" for index in range(message_count)]

    def split(*args, on_log, **kwargs):
        for message in messages:
            on_log(message)
        raise error_type("caption split failed")

    webui.grid_splitter.split_image = Mock(side_effect=split)

    status, gallery = webui.do_grid_split("grid.png", "Gemma-4-E2B", "без текста (только кадр)")

    assert status == "❌ caption split failed\n\n" + "\n".join(messages[-12:])
    assert gallery == []


def test_grid_split_logs_do_not_leak_between_requests(webui):
    def split(*args, on_log, **kwargs):
        on_log("old request log")
        raise RuntimeError("first failure")

    webui.grid_splitter.split_image = Mock(side_effect=split)
    assert "old request log" in webui.do_grid_split("first.png", "Qwen3-VL-4B", "с текстом (подписями)")[0]
    webui.grid_splitter.split_image.side_effect = ValueError("second failure")

    assert webui.do_grid_split("second.png", "Qwen3-VL-4B", "без текста (только кадр)") == (
        "❌ second failure\n\n", [],
    )


def test_grid_caption_dropdown_defaults_to_included_and_is_connected_to_split_handler(webui, monkeypatch, tmp_path):
    """Build the real UI against component mocks and exercise the registered click."""
    gradio = MagicMock()
    components = []

    def component(*args, **kwargs):
        result = MagicMock()
        result.__enter__.return_value = result
        components.append((args, kwargs, result))
        return result

    for name in (
        "Blocks", "Markdown", "Accordion", "Row", "Column", "Group", "Textbox", "Dropdown",
        "UploadButton", "Button", "Slider", "Checkbox", "Image", "File", "Gallery",
    ):
        getattr(gradio, name).side_effect = component
    monkeypatch.setattr(webui, "gr", gradio)
    monkeypatch.setattr(webui, "backend_status", Mock(return_value="offline"))
    monkeypatch.setattr(webui, "rag_core", Mock(
        CLOUD_PROVIDERS={}, load_settings=Mock(return_value={}),
        load_env=Mock(return_value={}), list_collections=Mock(return_value=[]),
    ))
    webui.grid_splitter.split_image = Mock(return_value=(tmp_path / "grid1", []))

    demo = webui.build_ui()

    assert demo is components[0][2]

    dropdowns = [
        (kwargs, widget) for _, kwargs, widget in components
        if kwargs.get("choices") == ("с текстом (подписями)", "без текста (только кадр)")
    ]
    assert len(dropdowns) == 1
    dropdown_config, dropdown = dropdowns[0]
    assert dropdown_config["value"] == "с текстом (подписями)"
    button = next(widget for args, _, widget in components if args == ("✂️ Gitter schneiden",))
    button.click.assert_called_once()
    binding = button.click.call_args.kwargs
    assert binding["fn"] is webui.do_grid_split
    image = next(widget for _, kwargs, widget in components if kwargs.get("label") == "🖼️ Gitter-Bild")
    model = next(widget for _, kwargs, widget in components if kwargs.get("label") == "Vision-Modell")
    assert binding["inputs"] == [image, model, dropdown]
    status = next(widget for _, kwargs, widget in components if kwargs.get("label") == "Status")
    gallery = next(widget for _, kwargs, widget in components if kwargs.get("label") == "Ergebnis")
    assert binding["outputs"] == [status, gallery]
    for choice, expected in [(dropdown_config["value"], True), ("без текста (только кадр)", False)]:
        binding["fn"]("grid.png", "Qwen3-VL-4B", choice)
        assert webui.grid_splitter.split_image.call_args.kwargs["with_captions"] is expected
