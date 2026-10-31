"""Tests for resolving per-camera audio sources for speaker detection."""

import pytest

from multicam_editor.logic.audio_plan import (
    AmbiguousSpeakerSourceError,
    AudioPlanError,
    resolve_camera_audio_plan,
    require_usable_alignment,
)
from multicam_editor.logic.audio_source_sync import AudioSourceAlignment
from multicam_editor.logic.audio_sources import AudioSource, CameraSpeakerAssignment


def _source(source_id, speaker_id=None, **kwargs):
    return AudioSource(
        source_id=source_id,
        path=f"/recordings/{source_id}.wav",
        speaker_id=speaker_id,
        **kwargs,
    )


def _alignment(source, offset_ms=0.0, status="ok", message="ok"):
    return AudioSourceAlignment(
        source_id=source.source_id,
        path=source.path,
        channel_index=source.channel_index,
        offset_ms=offset_ms,
        status=status,
        message=message,
    )


def test_maps_cameras_to_sources_with_their_offsets():
    travis = _source("travis", "Travis")
    dakota = _source("dakota", "Dakota")
    plan = resolve_camera_audio_plan(
        2,
        [travis, dakota],
        [CameraSpeakerAssignment(0, "Travis"), CameraSpeakerAssignment(1, "Dakota")],
        {
            "travis": _alignment(travis, 125.0),
            "dakota": _alignment(dakota, -40.0),
        },
    )

    assert [p.camera_index for p in plan] == [0, 1]
    assert plan[0].source is travis and plan[0].offset_ms == 125.0
    assert plan[1].source is dakota and plan[1].offset_ms == -40.0
    assert all(p.alignment_status == "ok" for p in plan)
    assert all(p.is_mapped and p.has_detection_audio for p in plan)


def test_explicitly_unmapped_and_unlisted_cameras_have_no_source():
    travis = _source("travis", "Travis")
    plan = resolve_camera_audio_plan(
        3,
        [travis],
        [CameraSpeakerAssignment(0, "Travis"), CameraSpeakerAssignment(1, None)],
        {"travis": _alignment(travis, 10.0)},
    )

    for wide in (plan[1], plan[2]):  # explicit None, and no entry at all
        assert wide.source is None
        assert wide.speaker_id is None
        assert wide.offset_ms == 0.0
        assert wide.alignment_status == "not_applicable"
        assert not wide.is_mapped and not wide.has_detection_audio


def test_speaker_without_any_source_is_an_error():
    with pytest.raises(AudioPlanError, match="Travis"):
        resolve_camera_audio_plan(
            1, [_source("dakota", "Dakota")], [CameraSpeakerAssignment(0, "Travis")]
        )


def test_several_detection_sources_for_one_speaker_is_ambiguous():
    lav = _source("travis_lav", "Travis")
    headset = _source("travis_headset", "Travis")
    with pytest.raises(AmbiguousSpeakerSourceError, match="Travis"):
        resolve_camera_audio_plan(
            1, [lav, headset], [CameraSpeakerAssignment(0, "Travis")]
        )


def test_preferred_source_resolves_ambiguity():
    lav = _source("travis_lav", "Travis")
    headset = _source("travis_headset", "Travis")
    plan = resolve_camera_audio_plan(
        1,
        [lav, headset],
        [CameraSpeakerAssignment(0, "Travis")],
        {"travis_headset": _alignment(headset, 5.0)},
        preferred_sources={"Travis": "travis_headset"},
    )
    assert plan[0].source is headset
    assert plan[0].offset_ms == 5.0


def test_preferred_source_must_belong_to_that_speaker():
    lav = _source("travis_lav", "Travis")
    other = _source("dakota", "Dakota")
    with pytest.raises(AudioPlanError):
        resolve_camera_audio_plan(
            1,
            [lav, other],
            [CameraSpeakerAssignment(0, "Travis")],
            preferred_sources={"Travis": "dakota"},
        )
    with pytest.raises(AudioPlanError):
        resolve_camera_audio_plan(
            1,
            [lav],
            [CameraSpeakerAssignment(0, "Travis")],
            preferred_sources={"Travis": "no_such_source"},
        )


def test_source_excluded_from_detection_leaves_camera_mapped_without_audio():
    output_only = _source("travis", "Travis", use_for_detection=False)
    plan = resolve_camera_audio_plan(
        1, [output_only], [CameraSpeakerAssignment(0, "Travis")]
    )
    assert plan[0].is_mapped
    assert plan[0].source is None
    assert not plan[0].has_detection_audio


def test_output_only_source_does_not_make_a_second_detection_source_ambiguous():
    lav = _source("travis_lav", "Travis")
    backup = _source("travis_backup", "Travis", use_for_detection=False)
    plan = resolve_camera_audio_plan(
        1, [lav, backup], [CameraSpeakerAssignment(0, "Travis")]
    )
    assert plan[0].source is lav


def test_two_cameras_may_share_one_speaker():
    travis = _source("travis", "Travis")
    plan = resolve_camera_audio_plan(
        2,
        [travis],
        [CameraSpeakerAssignment(0, "Travis"), CameraSpeakerAssignment(1, "Travis")],
        {"travis": _alignment(travis, 7.0)},
    )
    assert plan[0].source is travis and plan[1].source is travis


def test_invalid_camera_configuration_is_rejected():
    travis = _source("travis", "Travis")
    with pytest.raises(AudioPlanError):
        resolve_camera_audio_plan(
            2, [travis], [CameraSpeakerAssignment(2, "Travis")]
        )
    with pytest.raises(ValueError):
        resolve_camera_audio_plan(
            2,
            [travis],
            [CameraSpeakerAssignment(0, "Travis"), CameraSpeakerAssignment(0, None)],
        )
    with pytest.raises(AudioPlanError):
        resolve_camera_audio_plan(0, [], [])


def test_failed_and_missing_alignments_are_reported_not_hidden():
    travis = _source("travis", "Travis")
    dakota = _source("dakota", "Dakota")
    plan = resolve_camera_audio_plan(
        2,
        [travis, dakota],
        [CameraSpeakerAssignment(0, "Travis"), CameraSpeakerAssignment(1, "Dakota")],
        {"travis": _alignment(travis, 0.0, "failed", "Low correlation (0.001)")},
    )
    assert plan[0].alignment_status == "failed" and plan[0].offset_ms == 0.0
    assert plan[1].alignment_status == "missing" and plan[1].offset_ms == 0.0


def test_require_usable_alignment_names_every_unusable_mapped_source():
    travis = _source("travis", "Travis", label="Travis lav")
    dakota = _source("dakota", "Dakota")
    plan = resolve_camera_audio_plan(
        3,
        [travis, dakota],
        [
            CameraSpeakerAssignment(0, "Travis"),
            CameraSpeakerAssignment(1, "Dakota"),
            CameraSpeakerAssignment(2, None),
        ],
        {
            "travis": _alignment(travis, 0.0, "failed", "Low correlation"),
            "dakota": _alignment(dakota, 12.0),
        },
    )
    with pytest.raises(AudioPlanError) as excinfo:
        require_usable_alignment(plan)
    message = str(excinfo.value)
    assert "Travis lav" in message
    assert "dakota" not in message.lower()  # aligned source is not blamed


def test_require_usable_alignment_passes_when_everything_is_aligned():
    travis = _source("travis", "Travis")
    plan = resolve_camera_audio_plan(
        2,
        [travis],
        [CameraSpeakerAssignment(0, "Travis"), CameraSpeakerAssignment(1, None)],
        {"travis": _alignment(travis, 3.0)},
    )
    require_usable_alignment(plan)  # unmapped wide shot is not a problem
