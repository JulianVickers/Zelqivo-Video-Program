"""Tests for explicit timeline-boundary policy configuration."""

import pytest

from multicam_editor.logic.audio_sources import (
    TimelineBoundaryConfig,
    TimelineBoundaryPolicy,
    TimelinePlacementBasis,
    TimelineSourceSpan,
    build_preservation_timeline,
    map_sources_to_output_timeline,
)


def test_timeline_boundary_defaults_to_standard_behavior():
    config = TimelineBoundaryConfig()

    assert config.policy is TimelineBoundaryPolicy.STANDARD


def test_preserve_all_media_can_be_selected_with_enum():
    config = TimelineBoundaryConfig(TimelineBoundaryPolicy.PRESERVE_ALL_MEDIA)

    assert config.policy is TimelineBoundaryPolicy.PRESERVE_ALL_MEDIA


def test_preserve_all_media_can_be_selected_with_string():
    config = TimelineBoundaryConfig("preserve_all_media")

    assert config.policy is TimelineBoundaryPolicy.PRESERVE_ALL_MEDIA


def test_unknown_timeline_boundary_policy_is_rejected():
    with pytest.raises(ValueError, match="policy must be one of"):
        TimelineBoundaryConfig("trim_whatever_is_longer")


def test_preservation_timeline_covers_earliest_start_and_latest_end():
    timeline = build_preservation_timeline([
        TimelineSourceSpan(
            "camera", 0, 10_000, TimelinePlacementBasis.RECORDED_TIMESTAMP
        ),
        TimelineSourceSpan(
            "early_audio", -2_000, 5_000, TimelinePlacementBasis.USER_DEFINED
        ),
        TimelineSourceSpan(
            "late_audio", 8_000, 7_000, TimelinePlacementBasis.SYNC_ESTIMATE
        ),
    ])

    assert timeline.start_ms == -2_000
    assert timeline.end_ms == 15_000
    assert timeline.duration_ms == 17_000
    assert timeline.has_estimated_placements is True
    assert len(timeline.source_spans) == 3


def test_preservation_timeline_marks_fully_explicit_placements():
    timeline = build_preservation_timeline([
        TimelineSourceSpan(
            "camera", 0, 10_000, TimelinePlacementBasis.RECORDED_TIMESTAMP
        ),
        TimelineSourceSpan(
            "audio", 500, 8_000, TimelinePlacementBasis.USER_DEFINED
        ),
    ])

    assert timeline.has_estimated_placements is False
    assert timeline.duration_ms == 10_000


def test_preservation_timeline_rejects_empty_source_list():
    with pytest.raises(ValueError, match="at least one source"):
        build_preservation_timeline([])


def test_preservation_timeline_rejects_unknown_placement():
    spans = [
        TimelineSourceSpan("camera", 0, 10_000, TimelinePlacementBasis.USER_DEFINED),
        TimelineSourceSpan("unknown_audio", None, 8_000),
    ]

    with pytest.raises(ValueError, match="unknown source placement.*unknown_audio"):
        build_preservation_timeline(spans)


def test_preservation_timeline_rejects_duplicate_source_ids():
    spans = [
        TimelineSourceSpan("duplicate", 0, 10_000, TimelinePlacementBasis.USER_DEFINED),
        TimelineSourceSpan("duplicate", 1_000, 8_000, TimelinePlacementBasis.SYNC_ESTIMATE),
    ]

    with pytest.raises(ValueError, match="source IDs must be unique"):
        build_preservation_timeline(spans)


def test_timeline_span_rejects_invalid_duration_and_placement():
    with pytest.raises(ValueError, match="duration_ms must be positive"):
        TimelineSourceSpan("camera", 0, 0, TimelinePlacementBasis.USER_DEFINED)

    with pytest.raises(ValueError, match="requires UNKNOWN"):
        TimelineSourceSpan("audio", None, 1000, TimelinePlacementBasis.SYNC_ESTIMATE)

    with pytest.raises(ValueError, match="requires an explicit placement basis"):
        TimelineSourceSpan("audio", 0, 1000, TimelinePlacementBasis.UNKNOWN)


def test_timeline_span_end_can_be_negative_or_after_reference_end():
    early = TimelineSourceSpan("early", -5000, 2000, TimelinePlacementBasis.USER_DEFINED)
    late = TimelineSourceSpan("late", 12000, 3000, TimelinePlacementBasis.USER_DEFINED)

    assert early.end_ms == -3000
    assert late.end_ms == 15000


def test_timeline_placement_basis_accepts_serialized_value():
    span = TimelineSourceSpan("audio", 0, 1000, "sync_estimate")

    assert span.placement_basis is TimelinePlacementBasis.SYNC_ESTIMATE


def test_output_placements_shift_all_sources_to_zero_based_timeline():
    timeline = build_preservation_timeline([
        TimelineSourceSpan(
            "camera", 0, 30_000, TimelinePlacementBasis.USER_DEFINED
        ),
        TimelineSourceSpan(
            "early_audio", -2_000, 34_000,
            TimelinePlacementBasis.RECORDED_TIMESTAMP
        ),
        TimelineSourceSpan(
            "late_camera", 1_500, 28_000,
            TimelinePlacementBasis.SYNC_ESTIMATE
        ),
    ])

    placements = map_sources_to_output_timeline(timeline)
    by_id = {placement.source_id: placement for placement in placements}

    assert by_id["early_audio"].output_start_ms == 0
    assert by_id["early_audio"].output_end_ms == 34_000

    assert by_id["camera"].output_start_ms == 2_000
    assert by_id["camera"].output_end_ms == 32_000

    assert by_id["late_camera"].output_start_ms == 3_500
    assert by_id["late_camera"].output_end_ms == 31_500

    # Mapping retains each complete source duration and its timing provenance.
    assert by_id["camera"].duration_ms == 30_000
    assert (
        by_id["late_camera"].placement_basis
        is TimelinePlacementBasis.SYNC_ESTIMATE
    )


def test_output_placements_preserve_source_order():
    timeline = build_preservation_timeline([
        TimelineSourceSpan("later", 1000, 2000, TimelinePlacementBasis.USER_DEFINED),
        TimelineSourceSpan("earlier", -500, 1000, TimelinePlacementBasis.USER_DEFINED),
    ])

    placements = map_sources_to_output_timeline(timeline)

    assert [p.source_id for p in placements] == ["later", "earlier"]
    assert [p.output_start_ms for p in placements] == [1500, 0]


def test_output_placement_end_is_start_plus_full_duration():
    timeline = build_preservation_timeline([
        TimelineSourceSpan("audio", -1000, 5000, TimelinePlacementBasis.USER_DEFINED),
    ])

    placement, = map_sources_to_output_timeline(timeline)

    assert placement.output_start_ms == 0
    assert placement.duration_ms == 5000
    assert placement.output_end_ms == 5000
