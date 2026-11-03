"""Black (no-footage) segments in the single-pass render."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from multicam_editor.logic.video_merger import CutDefinition, render_single_pass
from multicam_editor.utils.ffmpeg import build_single_pass_filter_complex_args


def black(start, end, index, source_path=""):
    return CutDefinition(source_path, start, end, index, black=True)


def real(path, start, end, index, camera=0):
    return CutDefinition(str(path), start, end, index, camera_index=camera)


def build(cuts, **kwargs):
    """Run the real builder; return (args, filter script text or None)."""
    args, script_path = build_single_pass_filter_complex_args(cuts, "/output.mp4", **kwargs)
    text = None
    if script_path:
        try:
            with open(script_path, encoding="utf-8") as handle:
                text = handle.read()
        finally:
            os.remove(script_path)
    return args, text


def test_black_cut_adds_no_input_and_uses_a_color_source(tmp_path):
    cam = tmp_path / "cam.mp4"
    cam.write_text("fake")

    args, text = build([black(0, 2500, 0), real(cam, 2500, 6000, 1)])

    assert args.count("-i") == 1
    assert "color=c=black:s=1920x1080:r=30.0:d=2.500" in text
    assert "[0:v]trim=start=2.500:end=6.000" in text  # real source is still input 0
    assert "[v0][v1]concat=n=2:v=1:a=0" in text


def test_black_segment_ends_like_a_real_segment(tmp_path):
    cam = tmp_path / "cam.mp4"
    cam.write_text("fake")

    _, text = build([black(0, 2500, 0), real(cam, 2500, 6000, 1)])

    black_part = next(p for p in text.split(";") if "color=" in p)
    real_part = next(p for p in text.split(";") if "trim=" in p)
    assert "setsar=1,fps=30.0" in black_part
    assert "setsar=1,fps=30.0" in real_part


def test_black_follows_the_output_resolution(tmp_path):
    cam = tmp_path / "cam.mp4"
    cam.write_text("fake")

    _, text = build([black(0, 1000, 0), real(cam, 1000, 2000, 1)], resolution="720p")

    assert "color=c=black:s=1280x720" in text


def test_black_between_real_segments_keeps_input_numbering(tmp_path):
    cam0, cam1 = tmp_path / "cam0.mp4", tmp_path / "cam1.mp4"
    cam0.write_text("fake0")
    cam1.write_text("fake1")

    args, text = build([
        real(cam0, 0, 1000, 0, camera=0),
        black(1000, 2000, 1),
        real(cam1, 2000, 3000, 2, camera=1),
    ])

    assert args.count("-i") == 2
    assert "[0:v]" in text
    assert "[1:v]" in text
    assert "[2:v]" not in text
    assert "concat=n=3:v=1:a=0" in text


def test_black_cut_never_uses_its_source_path(tmp_path):
    cam = tmp_path / "cam.mp4"
    cam.write_text("fake")

    args, _ = build([black(0, 1000, 0, source_path="ignored.mp4"), real(cam, 1000, 2000, 1)])

    assert args.count("-i") == 1
    assert "ignored.mp4" not in args


def test_plan_with_only_black_segments_is_rejected():
    args, text = build([black(0, 5000, 0)])

    assert args == []
    assert text is None


def _fake_ffmpeg_process():
    proc = MagicMock()
    proc.run.return_value = MagicMock(success=True, cancelled=False, error=None)
    return proc


def test_render_single_pass_skips_the_file_check_for_black_cuts(tmp_path):
    cam = tmp_path / "cam.mp4"
    cam.write_text("fake")
    proc = _fake_ffmpeg_process()

    with patch("multicam_editor.logic.video_merger.is_ffmpeg_available", return_value=True), \
         patch("multicam_editor.logic.video_merger.FFmpegProcess", return_value=proc):
        result = render_single_pass(
            [black(0, 2000, 0), real(cam, 2000, 5000, 1)], str(tmp_path / "out.mp4")
        )

    assert result.success is True
    assert result.rendered_count == 2
    proc.run.assert_called_once()


def test_render_single_pass_still_rejects_a_missing_real_source(tmp_path):
    with patch("multicam_editor.logic.video_merger.is_ffmpeg_available", return_value=True):
        result = render_single_pass(
            [black(0, 2000, 0), real("/nope/missing.mp4", 2000, 5000, 1)],
            str(tmp_path / "out.mp4"),
        )

    assert result.success is False
    assert "missing.mp4" in result.error
