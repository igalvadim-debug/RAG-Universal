"""Caption trimming and its process, Python API, and CLI entry points (PR #9)."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock, call

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def splitter(monkeypatch):
    """Load the real splitter without starting a vision server."""
    spec = importlib.util.spec_from_file_location("grid_splitter", REPO_ROOT / "Workflow" / "grid_splitter.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "grid_splitter", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def panel():
    """An offset panel with an image, a wide gutter, and a caption band."""
    pixels = np.full((280, 280, 3), 255, dtype=np.uint8)
    pixels[40:160, 30:210] = (20, 60, 140)
    pixels[190:200, 50:190] = (180, 30, 20)
    return pixels, (20, 30, 220, 230)


def trim(splitter, pixels, box):
    """Exercise caption detection with real image-derived arrays."""
    image = Image.fromarray(pixels)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    return splitter.strip_caption(image, gray, splitter.bg_mask(image), box, *image.size)


def test_strip_caption_removes_separated_caption_without_changing_other_edges(splitter, panel):
    pixels, box = panel

    result = trim(splitter, pixels, box)

    assert result == (20, 30, 220, 190)
    # The whole picture survives, and none of the red caption pixels survives.
    cropped = pixels[result[1]:result[3], result[0]:result[2]]
    assert np.any(np.all(cropped == (20, 60, 140), axis=2))
    assert not np.any(np.all(cropped == (180, 30, 20), axis=2))


@pytest.mark.parametrize("size", [(39, 200), (200, 39), (39, 39), (40, 40)])
def test_strip_caption_preserves_small_panels(splitter, size):
    width, height = size
    pixels = np.full((height + 20, width + 20, 3), 255, dtype=np.uint8)
    box = (10, 10, 10 + width, 10 + height)

    assert trim(splitter, pixels, box) == box


@pytest.mark.parametrize("kind", ["blank", "solid", "picture-without-caption"])
def test_strip_caption_keeps_panels_without_a_plausible_caption(splitter, panel, kind):
    pixels, box = panel
    pixels[190:200, 50:190] = 255
    if kind == "blank":
        pixels[:] = 255
    elif kind == "solid":
        pixels[30:230, 20:220] = 0

    assert trim(splitter, pixels, box) == box


@pytest.mark.parametrize(
    ("caption_top", "expected_bottom"),
    [(38, 230), (78, 230), (98, 230), (100, 130), (160, 190)],
    ids=["below-absolute-minimum", "below-relative-minimum", "over-half-removed", "exactly-half", "normal"],
)
def test_strip_caption_rejects_implausible_height_but_accepts_half_height_boundary(
    splitter, panel, caption_top, expected_bottom,
):
    pixels, box = panel
    pixels[:] = 255
    pixels[30 + caption_top:40 + caption_top, 50:190] = 0

    assert trim(splitter, pixels, box) == (20, 30, 220, expected_bottom)


@pytest.mark.parametrize("gap", [1, 5, 14])
def test_strip_caption_treats_short_gaps_between_caption_lines_as_one_group(splitter, panel, gap):
    pixels, box = panel
    pixels[190:200, 50:190] = 255
    pixels[180:184, 50:190] = 0
    pixels[184 + gap:188 + gap, 50:190] = 0

    assert trim(splitter, pixels, box) == (20, 30, 220, 180)


@pytest.mark.parametrize(("noise_width", "expected_bottom"), [(2, 190), (3, 216)])
def test_strip_caption_row_content_threshold_ignores_sparse_noise(splitter, panel, noise_width, expected_bottom):
    pixels, box = panel
    # The inspected interior is 180 pixels wide: 2 pixels are below 1.5%, 3 above.
    pixels[216:219, 60:60 + noise_width] = 0

    assert trim(splitter, pixels, box) == (20, 30, 220, expected_bottom)


def test_strip_caption_ignores_frame_lines_in_the_excluded_margins(splitter, panel):
    pixels, box = panel
    pixels[30:230, 20:30] = 0
    pixels[30:230, 210:220] = 0
    pixels[220:230, 20:220] = 0

    assert trim(splitter, pixels, box) == (20, 30, 220, 190)


def test_strip_caption_does_not_modify_image_gray_or_background_mask(splitter, panel):
    pixels, box = panel
    image = Image.fromarray(pixels)
    original_image = image.tobytes()
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    mask = splitter.bg_mask(image)
    original_gray, original_mask = gray.copy(), mask.copy()
    gray.setflags(write=False)
    mask.setflags(write=False)

    assert splitter.strip_caption(image, gray, mask, box, *image.size) == (20, 30, 220, 190)

    assert image.tobytes() == original_image
    np.testing.assert_array_equal(gray, original_gray)
    np.testing.assert_array_equal(mask, original_mask)


@pytest.mark.parametrize(("edge", "expected_bottom"), [(-10, 110), (999, 280)])
def test_strip_caption_clamps_refined_edge_to_minimum_height_and_image_height(
    splitter, panel, monkeypatch, edge, expected_bottom,
):
    pixels, box = panel
    monkeypatch.setattr(splitter, "_edge", Mock(return_value=edge))

    assert trim(splitter, pixels, box) == (20, 30, 220, expected_bottom)


@pytest.fixture
def grid_input(splitter, monkeypatch, tmp_path):
    """Two captioned panels, with detections intentionally in reverse reading order."""
    pixels = np.full((240, 480, 3), 255, dtype=np.uint8)
    boxes = [(20, 20, 220, 220), (260, 20, 460, 220)]
    for (left, top, right, _), color in zip(boxes, [(20, 60, 140), (30, 140, 60)], strict=True):
        pixels[top + 10:top + 130, left + 10:right - 10] = color
        pixels[top + 160:top + 170, left + 30:right - 30] = (180, 30, 20)
    path = tmp_path / "storyboard.png"
    Image.fromarray(pixels).convert("RGBA").save(path)
    detections = [
        {"bbox_2d": [left / 480 * 1000, top / 240 * 1000, right / 480 * 1000, bottom / 240 * 1000]}
        for left, top, right, bottom in reversed(boxes)
    ]
    monkeypatch.setattr(splitter, "ask_model", Mock(return_value=(json.dumps(detections), (480, 240))))
    # Edge snapping predates this PR; keep these precise synthetic panel bounds.
    monkeypatch.setattr(splitter, "snap_box", lambda gray, mask, box, width, height: box)
    monkeypatch.setattr(splitter, "SCRIPT_DIR", tmp_path)
    return path, pixels, boxes


@pytest.mark.parametrize("captions", [None, True, False], ids=["default", "included", "trimmed"])
@pytest.mark.parametrize("debug", [False, True])
def test_process_applies_caption_choice_to_saved_panels_and_debug_image(
    splitter, grid_input, monkeypatch, tmp_path, captions, debug,
):
    path, pixels, boxes = grid_input
    strip = Mock(wraps=splitter.strip_caption)
    monkeypatch.setattr(splitter, "strip_caption", strip)
    options = {} if captions is None else {"with_captions": captions}

    folder, files = splitter.process(path, 8080, tmp_path / "output", debug, **options)

    assert folder == tmp_path / "output" / "grid1"
    assert len(files) == 2
    assert all(file.parent == folder for file in files)
    assert (folder.parent / "used.md").read_text(encoding="utf-8").startswith("grid1  # ")
    for file, (left, top, right, bottom) in zip(files, boxes, strict=True):
        if captions is False:
            bottom = top + 160
        with Image.open(file) as saved:
            assert saved.format == "PNG"
            assert saved.mode == "RGB"
            np.testing.assert_array_equal(np.asarray(saved), pixels[top:bottom, left:right])
    if captions is False:
        assert strip.call_count == 2
        assert [args.args[3] for args in strip.call_args_list] == boxes
    else:
        strip.assert_not_called()
    debug_path = tmp_path / "grid_splitter_debug.png"
    assert debug_path.exists() is debug
    if debug:
        with Image.open(debug_path) as preview:
            bottom = 180 if captions is False else 220
            assert preview.getpixel((25, bottom)) == (0, 255, 0)


def test_process_caption_failure_propagates_without_writing_output(splitter, grid_input, monkeypatch, tmp_path):
    path, _, _ = grid_input
    failure = ValueError("invalid caption data")
    monkeypatch.setattr(splitter, "strip_caption", Mock(side_effect=failure))
    out_root = tmp_path / "output"

    with pytest.raises(ValueError) as caught:
        splitter.process(path, 8080, out_root, False, with_captions=False)

    assert caught.value is failure
    assert not out_root.exists()


@pytest.fixture
def server_calls(splitter, monkeypatch):
    """Stub all server operations and processing at the API/CLI boundary."""
    calls = Mock()
    calls.server_has_vision.return_value = False
    for name in ("server_has_vision", "start_server", "stop_server", "process"):
        monkeypatch.setattr(splitter, name, getattr(calls, name))
    return calls


@pytest.mark.parametrize("captions", [None, True, False], ids=["default", "included", "trimmed"])
@pytest.mark.parametrize("reuse", [False, True], ids=["started-server", "existing-server"])
def test_split_image_forwards_caption_choice_and_preserves_server_lifecycle(
    splitter, server_calls, monkeypatch, tmp_path, captions, reuse,
):
    source = tmp_path / "grid.png"
    source.touch()
    out_root = tmp_path / "output"
    monkeypatch.setattr(splitter, "REUSE_PORTS", (1111, 2222))
    server_calls.server_has_vision.side_effect = [False, reuse]
    server_calls.process.return_value = (out_root / "grid1", [out_root / "grid1" / "1abc.png"])
    messages = []
    options = {} if captions is None else {"with_captions": captions}

    result = splitter.split_image(
        str(source), str(out_root), "gemma", 7, True, messages.append, **options,
    )

    assert result is server_calls.process.return_value
    server_calls.process.assert_called_once_with(
        source, 2222 if reuse else splitter.VL_PORT, out_root, True, captions is not False,
    )
    if reuse:
        server_calls.start_server.assert_not_called()
        server_calls.stop_server.assert_called_once_with(None)
        assert messages
    else:
        server_calls.start_server.assert_called_once_with(7, "gemma")
        server_calls.stop_server.assert_called_once_with(server_calls.start_server.return_value)
    assert splitter._log_cb is None


@pytest.mark.parametrize("reuse", [False, True])
def test_split_image_caption_processing_failure_cleans_up_and_allows_the_next_call(
    splitter, server_calls, tmp_path, reuse,
):
    source = tmp_path / "grid.png"
    source.touch()
    server_calls.server_has_vision.return_value = reuse
    failure = RuntimeError("caption processing failed")
    server_calls.process.side_effect = failure
    callback = Mock()

    with pytest.raises(RuntimeError) as caught:
        splitter.split_image(source, tmp_path / "output", on_log=callback, with_captions=False)

    assert caught.value is failure
    server_calls.stop_server.assert_called_once_with(None if reuse else server_calls.start_server.return_value)
    assert splitter._log_cb is None
    assert not splitter._run_lock.locked()
    callback.reset_mock()
    server_calls.process.side_effect = None

    assert splitter.split_image(source, tmp_path / "output", with_captions=True) is server_calls.process.return_value

    assert server_calls.process.call_args.args[-1] is True
    assert server_calls.stop_server.call_count == 2
    callback.assert_not_called()


@pytest.mark.parametrize("no_captions", [False, True])
@pytest.mark.parametrize("first_fails", [False, True])
def test_cli_forwards_caption_choice_for_every_image_and_remains_non_interactive(
    splitter, server_calls, monkeypatch, tmp_path, no_captions, first_fails,
):
    sources = [tmp_path / "first.png", tmp_path / "second.png"]
    for source in sources:
        source.touch()
    out_root = tmp_path / "output"
    argv = ["grid_splitter.py", *(str(source) for source in sources), "--out-root", str(out_root)]
    if no_captions:
        argv.append("--no-captions")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("PROMPT", "unit-test")
    monkeypatch.setattr("builtins.input", Mock(side_effect=AssertionError("CLI must not prompt")))
    if first_fails:
        server_calls.process.side_effect = [RuntimeError("first image failed"), None]

    with pytest.raises(SystemExit) as caught:
        splitter.main()

    assert caught.value.code == (1 if first_fails else 0)
    assert server_calls.process.call_args_list == [
        call(source, splitter.VL_PORT, out_root, False, with_captions=not no_captions)
        for source in sources
    ]
    server_calls.stop_server.assert_called_once_with(server_calls.start_server.return_value)
