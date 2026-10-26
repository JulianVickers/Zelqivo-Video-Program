"""Tests for explicit timeline-boundary policy configuration."""

import pytest

from multicam_editor.logic.audio_sources import (
    TimelineBoundaryConfig,
    TimelineBoundaryPolicy,
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
