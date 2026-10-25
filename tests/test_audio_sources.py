"""Tests for external audio-source and camera-assignment configuration."""

import pytest

from multicam_editor.logic.audio_sources import (
    AudioOutputConfig,
    AudioOutputMode,
    AudioSource,
    CameraSpeakerAssignment,
    validate_audio_sources,
    validate_camera_assignments,
)


def test_source_defaults_are_independent_and_mixdown_is_default():
    source = AudioSource(source_id="travis", path="/recordings/travis.wav")

    assert source.display_name == "travis.wav"
    assert source.speaker_id is None
    assert source.use_for_detection is True
    assert source.include_in_output is True
    assert source.gain_db == 0.0
    assert AudioOutputConfig().mode is AudioOutputMode.MIXDOWN


def test_channel_selection_is_zero_based_but_display_is_one_based():
    source = AudioSource(
        source_id="dakota-channel-3",
        path="/recordings/mixer.wav",
        channel_index=2,
    )

    assert source.channel_index == 2
    assert source.display_name == "mixer.wav — Channel 3"


def test_explicit_label_takes_precedence_over_derived_filename():
    source = AudioSource(
        source_id="speaker-a",
        path="/recordings/mixer.wav",
        channel_index=0,
        label="Travis — X-Air channel 1",
    )

    assert source.display_name == "Travis — X-Air channel 1"


def test_source_can_be_detection_only_or_output_only():
    detection_only = AudioSource(
        source_id="detect",
        path="detect.wav",
        use_for_detection=True,
        include_in_output=False,
    )
    output_only = AudioSource(
        source_id="mix",
        path="mix.wav",
        use_for_detection=False,
        include_in_output=True,
        gain_db=-6.0,
    )

    assert detection_only.use_for_detection and not detection_only.include_in_output
    assert not output_only.use_for_detection and output_only.include_in_output
    assert output_only.gain_db == -6.0


def test_source_ids_must_be_unique():
    sources = [
        AudioSource(source_id="same", path="one.wav"),
        AudioSource(source_id="same", path="two.wav"),
    ]

    with pytest.raises(ValueError, match="unique"):
        validate_audio_sources(sources)


def test_invalid_source_values_are_rejected():
    with pytest.raises(ValueError, match="source_id"):
        AudioSource(source_id=" ", path="audio.wav")
    with pytest.raises(ValueError, match="channel_index"):
        AudioSource(source_id="a", path="audio.wav", channel_index=-1)
    with pytest.raises(ValueError, match="finite"):
        AudioSource(source_id="a", path="audio.wav", gain_db=float("nan"))


def test_camera_may_be_explicitly_unmapped():
    wide_shot = CameraSpeakerAssignment(camera_index=2, speaker_id=None)

    assert wide_shot.camera_index == 2
    assert wide_shot.speaker_id is None


def test_camera_assignments_require_unique_camera_indices():
    assignments = [
        CameraSpeakerAssignment(camera_index=0, speaker_id="travis"),
        CameraSpeakerAssignment(camera_index=0, speaker_id=None),
    ]

    with pytest.raises(ValueError, match="unique camera indices"):
        validate_camera_assignments(assignments)


def test_output_mode_accepts_enum_or_serialized_value():
    assert AudioOutputConfig(AudioOutputMode.PRESERVE_TRACKS).mode is AudioOutputMode.PRESERVE_TRACKS
    assert AudioOutputConfig("preserve_tracks").mode is AudioOutputMode.PRESERVE_TRACKS

    with pytest.raises(ValueError, match="mode must be one of"):
        AudioOutputConfig("unknown")
