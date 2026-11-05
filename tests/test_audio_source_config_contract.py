"""Contract: mappings built by audio_source_config must resolve identically in the pipeline."""
import pytest

from multicam_editor.logic.audio_source_config import (
    AUTO_LABEL,
    SourceRow,
    check_camera_mapping,
    default_camera_mapping,
    default_speaker_names,
    rows_to_audio_sources,
    speaker_choices,
)
from multicam_editor.logic.pipeline_config import PipelineConfig
from multicam_editor.logic.processing_pipeline import ProcessingPipeline
from multicam_editor.utils.signals import ProcessingSignals

DIR = "/media/export/"
PAGECAST = [DIR + "session_Travis.wav", DIR + "session_Dakota.wav", DIR + "session_Julian.wav"]


def pagecast_rows():
    return [SourceRow(p, n) for p, n in zip(PAGECAST, default_speaker_names(PAGECAST))]


def resolve(rows, mapping):
    """Run the UI mapping through the real config and pipeline resolution."""
    pipeline = ProcessingPipeline(input_files=["a.mp4", "b.mp4"], signals=ProcessingSignals())
    pipeline._config = PipelineConfig.from_ui_mapping(mapping)
    pipeline._audio_sources = rows_to_audio_sources(rows, check_files=False)
    return pipeline._camera_sources_from_mapping()


def test_pagecast_filename_mapping_resolves_to_matching_files():
    cams = ["julian.mp4", "TRAVIS.mp4", "Dakota.mov", "wide.mp4"]
    rows = pagecast_rows()
    mapping = default_camera_mapping(cams, rows)
    assert check_camera_mapping(mapping, rows) == []
    assert PipelineConfig.from_ui_mapping(mapping).speaker_to_cameras_map == {2: [0], 0: [1], 1: [2]}
    assert resolve(rows, mapping) == {
        0: [PAGECAST[2]],
        1: [PAGECAST[0]],
        2: [PAGECAST[1]],
    }


def test_every_choice_value_resolves_to_its_own_speaker_files():
    rows = [SourceRow("/x/a.wav", "A"), SourceRow("/x/b.wav", "B"), SourceRow("/x/a_room.wav", "A", False)]
    for choice in speaker_choices(rows):
        got = resolve(rows, {0: choice.value})
        expected = [r.path for r in rows if r.speaker == choice.name]
        assert got == {0: expected}


def test_two_cameras_one_speaker():
    rows = pagecast_rows()
    mapping = default_camera_mapping(["Julian.mp4", "julian.mov", "wide.mp4"], rows)
    assert resolve(rows, mapping) == {0: [PAGECAST[2]], 1: [PAGECAST[2]]}


def test_noncontiguous_rows_number_by_first_appearance():
    rows = [SourceRow("/x/a.wav", "A"), SourceRow("/x/b.wav", "B"), SourceRow("/x/a_room.wav", "A", False)]
    assert resolve(rows, {0: "Speaker 1", 1: "Speaker 2"}) == {
        0: ["/x/a.wav", "/x/a_room.wav"],
        1: ["/x/b.wav"],
    }


def test_all_auto_resolves_to_none_and_is_flagged():
    rows = pagecast_rows()
    mapping = {0: AUTO_LABEL, 1: AUTO_LABEL}
    assert resolve(rows, mapping) is None
    assert check_camera_mapping(mapping, rows)


def test_unknown_speaker_flagged_here_and_raises_in_pipeline():
    rows = pagecast_rows()
    assert check_camera_mapping({0: "Speaker 4"}, rows)
    with pytest.raises(ValueError, match="Speaker 4"):
        resolve(rows, {0: "Speaker 4"})
