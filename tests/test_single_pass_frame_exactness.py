"""Rendered frame count must track the cut list, whatever the source frame rate.

Regression: per-segment ``fps`` rounding lost ~0.11 frame at every cut on sources
that are not exactly on the output grid (29.97, 23.976, VFR), so long cut lists
drifted against the session audio (0.593 s over 153 cuts on the PageCast render).
"""
import random
import shutil
import subprocess

import pytest

from multicam_editor.utils.ffmpeg import build_single_pass_filter_complex_args

FPS = 30.0
N_CUTS = 40

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not available",
)


def _make_cuts(src_path):
    rnd = random.Random(7)
    durs_ms = [rnd.randint(2200, 6000) for _ in range(N_CUTS)]
    cuts, t_ms = [], 5000
    for d in durs_ms:
        cuts.append({"source_path": src_path, "start_ms": t_ms, "end_ms": t_ms + d,
                     "camera_index": 0})
        t_ms += d + rnd.randint(0, 1000)
    return cuts, durs_ms, t_ms + 5000


def _frame_count(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


@pytest.mark.parametrize("rate", ["30000/1001", "30", "24000/1001"])
def test_rendered_frames_match_cut_list(tmp_path, rate):
    src = tmp_path / "src.mp4"
    cuts, durs_ms, need_ms = _make_cuts(str(src))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc=s=160x90:r={rate}:d={need_ms / 1000.0:.0f}", "-c:v", "libx264",
         "-preset", "ultrafast", str(src)], check=True)

    out = tmp_path / "out.mp4"
    args, script = build_single_pass_filter_complex_args(
        cuts, str(out), resolution="720p", fps=FPS)
    assert args, "no ffmpeg args produced"
    try:
        subprocess.run(args, check=True, capture_output=True)
    finally:
        if script:
            import os
            os.remove(script)

    expected = round(sum(durs_ms) / 1000.0 * FPS)
    actual = _frame_count(out)
    assert abs(actual - expected) <= 1, (
        f"source {rate}: {actual} frames, expected {expected} "
        f"({(actual - expected) / FPS * 1000:+.0f} ms)")
