"""Per-camera overrides for session placement: manual offset and exclude."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from multicam_editor.logic.session_placement import (
    CameraPlacement,
    SessionPlacement,
    SessionPlacementError,
    place_cameras_on_session,
)

_SP = "multicam_editor.logic.session_placement"
CAMS = ["/v/0.mp4", "/v/1.mp4", "/v/2.mp4"]
SRC = ["/rec/a.wav"]


def _ov(**kwargs):
    from multicam_editor.logic.session_placement import CameraOverride

    return CameraOverride(**kwargs)


@pytest.fixture
def env(monkeypatch):
    state = SimpleNamespace(
        align_calls=[],
        refine_calls=[],
        decoded=[],
        fail_decode=set(),
        refine_result=lambda coarse: SimpleNamespace(
            refined=True, offset_ms=coarse + 1.5, shift_ms=1.5, message=""
        ),
    )

    def decode(path):
        state.decoded.append(path)
        if path in state.fail_decode:
            raise RuntimeError("cannot decode")
        return np.zeros(16_000, dtype=np.float32)

    def fake_align(paths, sources, decode=None):
        state.align_calls.append(list(paths))
        out = []
        for position, path in enumerate(paths):
            decode(path)
            n = int(path.split("/")[-1].split(".")[0])
            out.append(SimpleNamespace(
                camera_index=position, path=path, status="ok",
                offset_ms=-1000.0 * (n + 1), peak=0.6, peak_ratio=12.0, message="",
            ))
        return out

    def fake_refine(camera_audio, session_mix, coarse_ms):
        state.refine_calls.append(coarse_ms)
        return state.refine_result(coarse_ms)

    monkeypatch.setattr(f"{_SP}.align_cameras_to_session", fake_align)
    monkeypatch.setattr(f"{_SP}.refine_offset", fake_refine)
    monkeypatch.setattr(f"{_SP}.mix_sources", lambda arrays: arrays[0])
    state.decode = decode
    return state


def _place(env, **kwargs):
    return place_cameras_on_session(CAMS, SRC, decode=env.decode, **kwargs)


def _camera(index, status):
    return CameraPlacement(
        camera_index=index, path=f"/v/{index}.mp4", status=status,
        session_offset_ms=0.0, slot_offset_ms=0.0, peak=0.5, peak_ratio=10.0,
        refined=False, refine_shift_ms=0.0, message="",
    )


# ---- CameraOverride --------------------------------------------------------

def test_override_rejects_exclude_with_manual_offset():
    with pytest.raises(ValueError, match="both"):
        _ov(exclude=True, manual_offset_ms=-1000.0)


def test_override_rejects_non_finite_offset():
    with pytest.raises(ValueError, match="finite"):
        _ov(manual_offset_ms=float("nan"))


# ---- manual offset ---------------------------------------------------------

def test_manual_camera_skips_coarse_match_and_keeps_index(env):
    result = _place(env, overrides={1: _ov(manual_offset_ms=-35110.0)})
    assert env.align_calls == [["/v/0.mp4", "/v/2.mp4"]]
    cams = result.cameras
    assert [c.camera_index for c in cams] == [0, 1, 2]
    assert [c.path for c in cams] == CAMS
    assert [c.status for c in cams] == ["ok", "manual", "ok"]
    assert cams[0].session_offset_ms == pytest.approx(-998.5)
    assert cams[2].session_offset_ms == pytest.approx(-2998.5)
    # the manual value is refined, then used
    assert -35110.0 in env.refine_calls
    assert cams[1].session_offset_ms == pytest.approx(-35108.5)
    assert cams[1].slot_offset_ms == pytest.approx(35108.5)
    assert cams[1].refined is True
    assert cams[1].peak == 0.0 and cams[1].peak_ratio == 0.0
    assert "manual" in cams[1].message.lower()


def test_manual_keeps_offset_when_refinement_rejected(env):
    env.refine_result = lambda c: SimpleNamespace(
        refined=False, offset_ms=c, shift_ms=0.0, message="spread too large"
    )
    result = _place(env, overrides={1: _ov(manual_offset_ms=-35110.0)})
    cam = result.cameras[1]
    assert cam.status == "manual"
    assert cam.session_offset_ms == -35110.0
    assert cam.refined is False
    assert any("1.mp4" in w and "spread too large" in w and "manual" in w for w in result.warnings)


def test_manual_keeps_offset_when_camera_cannot_be_decoded(env):
    env.fail_decode.add("/v/1.mp4")
    result = _place(env, overrides={1: _ov(manual_offset_ms=-35110.0)})
    cam = result.cameras[1]
    assert cam.status == "manual"
    assert cam.session_offset_ms == -35110.0
    assert cam.slot_offset_ms == 35110.0
    assert cam.refined is False
    assert -35110.0 not in env.refine_calls
    assert [w for w in result.warnings if "1.mp4" in w]


def test_all_manual_cameras_need_no_coarse_match(env):
    overrides = {i: _ov(manual_offset_ms=-1000.0 * (i + 1)) for i in range(3)}
    result = _place(env, overrides=overrides)
    assert env.align_calls == []
    assert [c.status for c in result.cameras] == ["manual"] * 3
    assert env.refine_calls == [-1000.0, -2000.0, -3000.0]


# ---- exclude ---------------------------------------------------------------

def test_excluded_camera_is_not_decoded_or_matched(env):
    result = _place(env, overrides={1: _ov(exclude=True)})
    assert env.align_calls == [["/v/0.mp4", "/v/2.mp4"]]
    assert "/v/1.mp4" not in env.decoded
    cams = result.cameras
    assert [c.camera_index for c in cams] == [0, 1, 2]
    assert [c.status for c in cams] == ["ok", "excluded", "ok"]
    assert cams[1].session_offset_ms == 0.0
    assert str(cams[1].slot_offset_ms) == "0.0"
    assert cams[1].refined is False


def test_all_cameras_excluded_is_an_error(env):
    overrides = {i: _ov(exclude=True) for i in range(3)}
    with pytest.raises(SessionPlacementError, match="excluded"):
        _place(env, overrides=overrides)


def test_override_for_unknown_camera_is_an_error(env):
    with pytest.raises(SessionPlacementError, match="7"):
        _place(env, overrides={7: _ov(exclude=True)})


# ---- what counts as placed -------------------------------------------------

def test_manual_and_excluded_count_as_placed():
    placement = SessionPlacement(
        cameras=(_camera(0, "ok"), _camera(1, "manual"), _camera(2, "excluded")),
        session_duration_ms=60_000,
    )
    assert placement.failures == ()
    placement.require_all_placed()


def test_failed_and_unreliable_still_fail():
    placement = SessionPlacement(
        cameras=(_camera(0, "ok"), _camera(1, "failed"), _camera(2, "unreliable")),
        session_duration_ms=60_000,
    )
    assert [c.camera_index for c in placement.failures] == [1, 2]
    with pytest.raises(SessionPlacementError):
        placement.require_all_placed()


def test_no_overrides_aligns_every_camera(env):
    result = _place(env)
    assert env.align_calls == [CAMS]
    assert [c.status for c in result.cameras] == ["ok"] * 3
