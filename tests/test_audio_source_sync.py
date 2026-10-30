"""Tests for independent synchronization of configured audio sources."""

from pathlib import Path

from multicam_editor.logic.audio_sources import AudioSource
from multicam_editor.logic import audio_source_sync


def _source(source_id: str, channel_index: int | None = None) -> AudioSource:
    return AudioSource(
        source_id=source_id,
        path=f"/recordings/{source_id}.wav",
        channel_index=channel_index,
    )


def test_aligns_multiple_sources_independently_and_preserves_channel_selection(
    monkeypatch,
    tmp_path: Path,
):
    extraction_calls = []
    alignment_calls = []

    def fake_extract(
        source_path,
        output_path=None,
        sample_rate=16000,
        mono=True,
        timeout=60.0,
        channel_index=None,
    ):
        extraction_calls.append((source_path, channel_index, output_path))
        Path(output_path).write_bytes(b"temporary wav")
        return type("Result", (), {"success": True, "error": None})()

    def fake_align(extracted_path, reference_path):
        alignment_calls.append((extracted_path, reference_path))
        return (125.0, "ok")

    monkeypatch.setattr(audio_source_sync, "extract_audio_to_wav", fake_extract)
    monkeypatch.setattr(audio_source_sync, "align_audio_offset", fake_align)

    sources = [
        _source("travis", channel_index=0),
        _source("dakota", channel_index=1),
    ]

    results = audio_source_sync.align_audio_sources(sources, "/reference.wav")

    assert set(results) == {"travis", "dakota"}
    assert results["travis"].status == "ok"
    assert results["dakota"].status == "ok"
    assert results["travis"].offset_ms == 125.0
    assert results["dakota"].offset_ms == 125.0

    assert [call[1] for call in extraction_calls] == [0, 1]
    assert all(call[1] == "/reference.wav" for call in alignment_calls)
    assert all(not Path(call[2]).exists() for call in extraction_calls)


def test_low_confidence_alignment_is_recorded_as_failure(monkeypatch):
    def fake_extract(source_path, output_path=None, **kwargs):
        Path(output_path).write_bytes(b"temporary wav")
        return type("Result", (), {"success": True, "error": None})()

    monkeypatch.setattr(audio_source_sync, "extract_audio_to_wav", fake_extract)
    monkeypatch.setattr(
        audio_source_sync,
        "align_audio_offset",
        lambda extracted, reference: (0.0, "Low correlation (0.001); alignment is unreliable"),
    )

    results = audio_source_sync.align_audio_sources(
        [_source("travis")],
        "/reference.wav",
    )

    result = results["travis"]
    assert result.status == "failed"
    assert result.offset_ms == 0.0
    assert "correlation" in result.message.lower()


def test_extraction_failure_does_not_stop_later_sources(monkeypatch):
    calls = []

    def fake_extract(source_path, output_path=None, **kwargs):
        calls.append(source_path)
        if source_path.endswith("travis.wav"):
            return type("Result", (), {
                "success": False,
                "error": "simulated extraction failure",
            })()

        Path(output_path).write_bytes(b"temporary wav")
        return type("Result", (), {"success": True, "error": None})()

    monkeypatch.setattr(audio_source_sync, "extract_audio_to_wav", fake_extract)
    monkeypatch.setattr(
        audio_source_sync,
        "align_audio_offset",
        lambda extracted, reference: (42.0, "ok"),
    )

    results = audio_source_sync.align_audio_sources(
        [_source("travis"), _source("dakota")],
        "/reference.wav",
    )

    assert calls == [
        "/recordings/travis.wav",
        "/recordings/dakota.wav",
    ]
    assert results["travis"].status == "failed"
    assert "simulated extraction failure" in results["travis"].message
    assert results["dakota"].status == "ok"
    assert results["dakota"].offset_ms == 42.0


def test_unexpected_error_is_isolated_to_its_source(monkeypatch):
    def fake_extract(source_path, output_path=None, **kwargs):
        if source_path.endswith("travis.wav"):
            raise RuntimeError("simulated unexpected error")

        Path(output_path).write_bytes(b"temporary wav")
        return type("Result", (), {"success": True, "error": None})()

    monkeypatch.setattr(audio_source_sync, "extract_audio_to_wav", fake_extract)
    monkeypatch.setattr(
        audio_source_sync,
        "align_audio_offset",
        lambda extracted, reference: (10.0, "ok"),
    )

    results = audio_source_sync.align_audio_sources(
        [_source("travis"), _source("dakota")],
        "/reference.wav",
    )

    assert results["travis"].status == "failed"
    assert "simulated unexpected error" in results["travis"].message
    assert results["dakota"].status == "ok"
