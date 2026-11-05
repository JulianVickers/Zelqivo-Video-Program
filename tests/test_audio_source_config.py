"""Tests for the pure-logic audio-source mapping helpers (no Qt)."""
import pytest

from multicam_editor.logic.audio_source_config import (
    AUTO_LABEL,
    SourceConfigError,
    SourceRow,
    check_camera_mapping,
    default_camera_mapping,
    default_speaker_names,
    match_cameras_to_speakers,
    rows_to_audio_sources,
    speaker_choices,
    speaker_order,
    validate_rows,
)

PAGECAST = ["session_Travis.wav", "session_Dakota.wav", "session_Julian.wav"]


def row(path, name, det=True, out=True, gain=0.0):
    return SourceRow(path, name, det, out, gain)


def pagecast_rows():
    names = default_speaker_names(PAGECAST)
    return [row(p, n) for p, n in zip(PAGECAST, names)]


class TestDefaultSpeakerNames:
    def test_pagecast_prefix_stripped(self):
        assert default_speaker_names(PAGECAST) == ["Travis", "Dakota", "Julian"]

    def test_no_shared_prefix(self):
        assert default_speaker_names(["Travis.wav", "Dakota.wav"]) == ["Travis", "Dakota"]

    def test_prefix_leaving_empty_name_refused(self):
        assert default_speaker_names(["session_.wav", "session_x.wav"]) == ["session_", "session_x"]

    def test_prefix_leaving_duplicate_refused(self):
        assert default_speaker_names(["s_a.wav", "s_a.wav"]) == ["s_a", "s_a"]

    def test_single_file_keeps_stem(self):
        assert default_speaker_names(["/x/session_Julian.wav"]) == ["session_Julian"]

    def test_dash_separator(self):
        assert default_speaker_names(["ep-Travis.wav", "ep-Dakota.wav"]) == ["Travis", "Dakota"]

    def test_numbered_files_quirk(self):
        assert default_speaker_names(["Julian_1.wav", "Julian_2.wav"]) == ["1", "2"]


class TestSpeakerOrderAndChoices:
    def test_first_appearance_and_blank_ignored(self):
        rows = [row("a.wav", "B"), row("b.wav", "  "), row("c.wav", "A"), row("d.wav", "B")]
        assert speaker_order(rows) == ["B", "A"]

    def test_choices_values_labels_and_detection(self):
        rows = [row("a.wav", "Travis"), row("b.wav", "Dakota", det=False), row("c.wav", "Travis", det=False)]
        ch = speaker_choices(rows)
        assert [c.value for c in ch] == ["Speaker 1", "Speaker 2"]
        assert [c.label for c in ch] == ["Speaker 1 - Travis", "Speaker 2 - Dakota"]
        assert [c.has_detection for c in ch] == [True, False]


class TestValidateRows:
    def test_empty_is_valid(self):
        assert validate_rows([]) == []

    def test_valid_pagecast(self):
        assert validate_rows(pagecast_rows(), check_files=False) == []

    def test_blank_speaker(self):
        assert any("speaker name is empty" in p for p in validate_rows([row("a.wav", " ")], check_files=False))

    def test_case_variant_names(self):
        probs = validate_rows([row("a.wav", "Julian"), row("b.wav", "julian", det=False)], check_files=False)
        assert any("differ only by case" in p for p in probs)

    def test_duplicate_path(self):
        probs = validate_rows([row("a.wav", "A"), row("a.wav", "B")], check_files=False)
        assert any("same file as row 1" in p for p in probs)

    def test_two_detection_sources_one_speaker(self):
        probs = validate_rows([row("a.wav", "A"), row("b.wav", "A")], check_files=False)
        assert any("2 sources used for detection" in p for p in probs)

    def test_lav_plus_output_only_room_mic_ok(self):
        rows = [row("lav.wav", "A"), row("room.wav", "A", det=False)]
        assert validate_rows(rows, check_files=False) == []

    def test_none_for_detection(self):
        probs = validate_rows([row("a.wav", "A", det=False)], check_files=False)
        assert any("At least one source must be used for detection" in p for p in probs)

    def test_missing_file_checked(self, tmp_path):
        probs = validate_rows([row(str(tmp_path / "nope.wav"), "A")], check_files=True)
        assert any("file not found" in p for p in probs)

    def test_existing_file_passes(self, tmp_path):
        f = tmp_path / "a.wav"
        f.write_bytes(b"")
        assert validate_rows([row(str(f), "A")], check_files=True) == []

    def test_non_finite_gain(self):
        probs = validate_rows([row("a.wav", "A", gain=float("nan"))], check_files=False)
        assert any("gain" in p for p in probs)


class TestRowsToAudioSources:
    def test_fields_and_ids(self):
        rows = [row("/x/a.wav", "A", out=False, gain=-3.0), row("/x/b.wav", "B")]
        src = rows_to_audio_sources(rows, check_files=False)
        assert [s.source_id for s in src] == ["src1", "src2"]
        assert src[0].path == "/x/a.wav" and src[0].label == "a.wav"
        assert src[0].speaker_id == "A" and src[0].channel_index is None
        assert src[0].include_in_output is False and src[0].gain_db == -3.0
        assert src[1].use_for_detection is True

    def test_invalid_rows_raise(self):
        with pytest.raises(SourceConfigError):
            rows_to_audio_sources([row("a.wav", "A"), row("b.wav", "A")], check_files=False)

    def test_speaker_name_stripped(self):
        src = rows_to_audio_sources([row("a.wav", "  A  ")], check_files=False)
        assert src[0].speaker_id == "A"


class TestCameraMatching:
    CAMS = ["julian.mp4", "TRAVIS.mp4", "Dakota.mov", "wide.mp4"]

    def test_pagecast_mapping(self):
        assert default_camera_mapping(self.CAMS, pagecast_rows()) == {
            0: "Speaker 3", 1: "Speaker 1", 2: "Speaker 2", 3: AUTO_LABEL,
        }

    def test_match_by_file_stem(self):
        rows = [row("/x/cam_a.wav", "Alice")]
        m = match_cameras_to_speakers(["/y/CAM_A.mp4"], rows)
        assert m[0].speaker_name == "Alice" and m[0].matched_file == "cam_a.wav"

    def test_match_by_speaker_name(self):
        m = match_cameras_to_speakers(["alice.mp4"], [row("/x/mic1.wav", "Alice")])
        assert m[0].speaker_name == "Alice" and m[0].matched_file is None

    def test_ambiguous_stays_auto_and_flagged(self):
        rows = [row("/a/Alex.wav", "A"), row("/b/Alex.wav", "B")]
        m = match_cameras_to_speakers(["alex.mp4"], rows)
        assert m[0].ambiguous and m[0].speaker_name is None
        assert default_camera_mapping(["alex.mp4"], rows) == {0: AUTO_LABEL}

    def test_unmatched_not_ambiguous(self):
        m = match_cameras_to_speakers(["wide.mp4"], pagecast_rows())
        assert not m[0].ambiguous and m[0].speaker_name is None

    def test_two_cameras_one_speaker(self):
        rows = [row("/x/Julian.wav", "Julian")]
        assert default_camera_mapping(["Julian.mp4", "julian.mov"], rows) == {0: "Speaker 1", 1: "Speaker 1"}

    def test_no_fuzzy_contains(self):
        assert default_camera_mapping(["julian_closeup.mp4"], pagecast_rows()) == {0: AUTO_LABEL}


class TestCheckCameraMapping:
    def test_no_rows_no_problems(self):
        assert check_camera_mapping({0: AUTO_LABEL}, []) == []

    def test_all_auto(self):
        probs = check_camera_mapping({0: AUTO_LABEL, 1: AUTO_LABEL}, pagecast_rows())
        assert probs == ["Map at least one camera to a speaker"]

    def test_valid(self):
        assert check_camera_mapping({0: "Speaker 1", 1: AUTO_LABEL}, pagecast_rows()) == []

    def test_unknown_speaker(self):
        probs = check_camera_mapping({0: "Speaker 9"}, pagecast_rows())
        assert any("Speaker 9 does not exist" in p for p in probs)

    def test_unrecognised_value(self):
        probs = check_camera_mapping({0: "Spekaer 1"}, pagecast_rows())
        assert any("unrecognised" in p for p in probs)

    def test_speaker_without_detection_source(self):
        rows = [row("a.wav", "A"), row("b.wav", "B", det=False)]
        probs = check_camera_mapping({0: "Speaker 2"}, rows)
        assert any("no source enabled for detection" in p for p in probs)
