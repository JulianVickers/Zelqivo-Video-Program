"""Pure-logic helpers behind the audio-sources mapping UI (no Qt).

The pipeline resolves "Speaker N" as the Nth distinct speaker_id by first
appearance among the configured sources, so row order defines the numbering.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .audio_sources import AudioSource, validate_audio_sources

AUTO_LABEL = "Auto (best effort)"
_SPEAKER_RE = re.compile(r"^Speaker (\d+)$")


class SourceConfigError(ValueError):
    """Raised when dialog rows cannot be turned into audio sources."""


@dataclass(frozen=True)
class SourceRow:
    path: str
    speaker_name: str
    use_for_detection: bool = True
    include_in_output: bool = True
    gain_db: float = 0.0

    @property
    def speaker(self) -> str:
        return self.speaker_name.strip()


@dataclass(frozen=True)
class SpeakerChoice:
    value: str  # "Speaker N": what the pipeline receives
    label: str  # "Speaker N - Name": what the combo shows
    name: str
    has_detection: bool


@dataclass(frozen=True)
class CameraMatch:
    speaker_name: str | None
    ambiguous: bool = False
    matched_file: str | None = None  # file name when matched by file stem


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].strip()


def default_speaker_names(paths: Sequence[str]) -> list[str]:
    """Suggest speaker names from file names, stripping a shared prefix."""
    stems = [_stem(p) for p in paths]
    if len(stems) < 2:
        return stems
    common = os.path.commonprefix(stems)
    cut = max(common.rfind("_"), common.rfind("-"))
    if cut < 0:
        return stems
    stripped = [s[cut + 1:].strip() for s in stems]
    if any(not s for s in stripped):
        return stems
    if len({s.casefold() for s in stripped}) != len(stripped):
        return stems
    return stripped


def speaker_order(rows: Sequence[SourceRow]) -> list[str]:
    """Distinct non-blank speaker names by first appearance."""
    seen: list[str] = []
    for row in rows:
        name = row.speaker
        if name and name not in seen:
            seen.append(name)
    return seen


def speaker_choices(rows: Sequence[SourceRow]) -> list[SpeakerChoice]:
    choices = []
    for n, name in enumerate(speaker_order(rows), start=1):
        has_det = any(r.speaker == name and r.use_for_detection for r in rows)
        choices.append(SpeakerChoice(f"Speaker {n}", f"Speaker {n} - {name}", name, has_det))
    return choices


def validate_rows(rows: Sequence[SourceRow], check_files: bool = True) -> list[str]:
    """User-facing problems; an empty list means valid (no rows = feature off)."""
    if not rows:
        return []
    problems: list[str] = []
    seen_paths: dict[str, int] = {}
    for i, row in enumerate(rows, start=1):
        tag = f"Row {i}"
        has_path = bool(row.path.strip())
        if not has_path:
            problems.append(f"{tag}: no file chosen")
        elif check_files and not os.path.isfile(row.path):
            problems.append(f"{tag}: file not found: {row.path}")
        if not row.speaker:
            problems.append(f"{tag}: speaker name is empty")
        if not math.isfinite(row.gain_db):
            problems.append(f"{tag}: gain must be a finite number")
        if has_path:
            key = os.path.normcase(os.path.abspath(row.path))
            if key in seen_paths:
                problems.append(f"{tag}: same file as row {seen_paths[key]}")
            else:
                seen_paths[key] = i
    names = speaker_order(rows)
    folded: dict[str, list[str]] = {}
    for name in names:
        folded.setdefault(name.casefold(), []).append(name)
    for group in folded.values():
        if len(group) > 1:
            problems.append("Speaker names differ only by case: " + ", ".join(group))
    if not any(r.use_for_detection for r in rows):
        problems.append("At least one source must be used for detection")
    for name in names:
        n_det = sum(1 for r in rows if r.speaker == name and r.use_for_detection)
        if n_det > 1:
            problems.append(
                f"Speaker '{name}' has {n_det} sources used for detection; "
                "keep exactly one (others can be output-only)"
            )
    return problems


def rows_to_audio_sources(rows: Sequence[SourceRow], check_files: bool = True) -> list[AudioSource]:
    problems = validate_rows(rows, check_files=check_files)
    if problems:
        raise SourceConfigError("; ".join(problems))
    sources = [
        AudioSource(
            source_id=f"src{i}",
            path=row.path,
            label=os.path.basename(row.path),
            channel_index=None,
            speaker_id=row.speaker,
            use_for_detection=row.use_for_detection,
            include_in_output=row.include_in_output,
            gain_db=row.gain_db,
        )
        for i, row in enumerate(rows, start=1)
    ]
    try:
        validate_audio_sources(sources)
    except ValueError as exc:
        raise SourceConfigError(str(exc)) from exc
    return sources


def match_cameras_to_speakers(
    camera_paths: Sequence[str], rows: Sequence[SourceRow]
) -> dict[int, CameraMatch]:
    """Match cameras to speakers by exact (case-folded) stem; no fuzzy matching."""
    names = speaker_order(rows)
    result: dict[int, CameraMatch] = {}
    for idx, cam in enumerate(camera_paths):
        stem = _stem(cam).casefold()
        hits: list[tuple[str, str | None]] = []
        for name in names:
            file_hit = next(
                (os.path.basename(r.path) for r in rows
                 if r.speaker == name and _stem(r.path).casefold() == stem),
                None,
            )
            if file_hit is not None or name.casefold() == stem:
                hits.append((name, file_hit))
        if len(hits) == 1:
            result[idx] = CameraMatch(hits[0][0], False, hits[0][1])
        elif len(hits) > 1:
            result[idx] = CameraMatch(None, True, None)
        else:
            result[idx] = CameraMatch(None, False, None)
    return result


def default_camera_mapping(camera_paths: Sequence[str], rows: Sequence[SourceRow]) -> dict[int, str]:
    numbers = {name: n for n, name in enumerate(speaker_order(rows), start=1)}
    mapping: dict[int, str] = {}
    for idx, match in match_cameras_to_speakers(camera_paths, rows).items():
        if match.speaker_name is None:
            mapping[idx] = AUTO_LABEL
        else:
            mapping[idx] = f"Speaker {numbers[match.speaker_name]}"
    return mapping


def check_camera_mapping(mapping: Mapping[int, str], rows: Sequence[SourceRow]) -> list[str]:
    """Pre-run problems the pipeline would otherwise report late."""
    if not rows:
        return []
    problems: list[str] = []
    choices = speaker_choices(rows)
    mapped: set[int] = set()
    for cam, value in mapping.items():
        if value.startswith("Auto"):
            continue
        m = _SPEAKER_RE.match(value)
        if not m:
            problems.append(f"Camera {cam + 1}: unrecognised speaker '{value}'")
            continue
        n = int(m.group(1))
        if n < 1 or n > len(choices):
            problems.append(f"Camera {cam + 1}: Speaker {n} does not exist")
            continue
        mapped.add(n)
    if not mapped and not problems:
        problems.append("Map at least one camera to a speaker")
    for n in sorted(mapped):
        choice = choices[n - 1]
        if not choice.has_detection:
            problems.append(f"{choice.label} has no source enabled for detection")
    return problems
