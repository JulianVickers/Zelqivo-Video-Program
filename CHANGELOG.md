# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Session placement now refines every camera's offset against the session mix
  (waveform refinement); if the refinement windows disagree by more than 25 ms
  the coarse offset is kept and a warning is logged.
- **License changed from MIT to Apache-2.0**, with the real copyright holder
  (Roman Berlin) replacing the "Owner" placeholder. Apache-2.0 adds an explicit
  patent grant and trademark reservation, which the open-core model needs.
- README rewritten as a user-first landing page: what it does / doesn't do,
  how it works, platform table, install guide, CLI reference, honest CI note,
  and the new logo. Above the fold: a demo GIF produced by the real pipeline
  on synthetic test cameras (`docs/assets/demo.gif`). Also removed the false
  "CI ready" claim. A social-preview banner was added under
  `Installer/assets/icons/`.
- Internal AI-agent scaffolding (`.clinerules`, `.ai_instructions/`,
  `.human_instructions/`) replaced by a single public `CLAUDE.md` with the
  project's engineering rules.
- Migrated the entire GUI layer from PyQt6 (GPLv3) to PySide6 ≥ 6.8 (LGPLv3):
  34 files, 80 import lines, 37 signal declarations (`pyqtSignal` → `Signal`).
  Unblocks the open-core licensing model (Phase 0.1 of the hardening work order).
- `pyproject.toml` is now the single source of truth for dependencies;
  `pytest-qt` added to the dev extras so the Qt test files run in the standard suite.

### Added
- Multi-source audio in the GUI: an "Audio sources…" dialog (one file per
  speaker, speaker name, detection and output flags, gain, reorder, live
  validation) can be used instead of the single external audio file. Camera
  speaker combos are built from the configured speakers and prefilled by
  matching camera and audio file names; hand edits are kept. Before
  processing, the app asks to switch to Fast for that run when another
  strategy is selected (the saved setting is untouched) and stops on camera
  mapping problems.
- **Multi-source session timeline** (backend only; the mapping UI is not wired
  yet): external audio sources (e.g. mixer exports) and cameras are placed on one
  shared session timeline. The mixer session is the ruler; the output always spans
  the whole session, with black video where no camera has footage and silence
  where there is no audio. A camera that cannot be placed stops the run with an
  error naming the camera and the numbers involved.
- Per-camera `manual_offset_ms` and `exclude` overrides as escape hatches for
  session placement, and the session audio mix is muxed in the concat stage.
- **Speaker detection on audio sources**: with configured sources, active
  speakers are detected on the clean per-speaker tracks instead of camera
  audio. Fast (Energy) backend only; Hybrid and Lips are refused for now.
- **Wide-shot pass** for a camera that is not mapped to any speaker, fully
  automatic: cut to the wide shot after 2 s of nobody speaking, and a 3 s
  cutaway after a 20 s hold on one camera. Cutaways only land inside the wide
  camera's footage. It runs after the decision engine and never changes it.
- **Runtime H.264 encoder selection**: hardware encoders (NVIDIA NVENC, Intel
  QSV, AMD AMF, Apple VideoToolbox) are detected and preferred automatically,
  with `libopenh264`/`libx264` as software fallbacks — replacing six hardcoded
  `libx264` sites. The chosen encoder is logged and recorded in the QA
  artifacts summary.
- **"Use my own FFmpeg" setting** (Settings → FFmpeg): point Zelqivo at an
  ffmpeg binary you installed yourself; ffprobe is found next to it. Takes
  effect without a restart.
- **Continuous integration**: GitHub Actions now run the full test suite
  (Qt included) on Linux, Windows, and macOS across Python 3.10–3.12, plus an
  advisory lint job; a release workflow builds the Windows installer from a
  version tag using an LGPL FFmpeg build, with SHA-256 checksums.
- Real app screenshots under `docs/screenshots/`, including a branded hero shot
  in the README.
- `tests/test_settings_roundtrip.py`: proves `QSettings.value(..., type=X)` coercion
  for bool/int/float/str plus missing-key defaults, guarding future PySide6 bumps.
- Open-source community files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
  (Contributor Covenant 2.1), `SECURITY.md`, `NOTICE`, `TRADEMARK.md`,
  GitHub issue templates and a pull-request template, `docs/THIRD_PARTY.md`,
  `docs/ROADMAP.md`, and `docs/GOOD_FIRST_ISSUES.md`.
- `src/multicam_editor/ui/widgets/__init__.py` — the widgets folder was an
  implicit namespace package, which breaks some packaging tools.
- New app logo (`Installer/assets/icons/`): PNG plus a multi-size `.ico`,
  now committed so contributors can build the installer; the PyInstaller
  spec uses it as the EXE icon.

- When `libx264` is the selected encoder, all renders now use CRF 18 explicitly
  (four call sites previously inherited FFmpeg's default CRF 23) — slightly
  larger intermediates, better final quality.
- Windows build scripts (`build_exe.ps1`, `multicam_editor.spec`) read the
  FFmpeg location from the `ZELQIVO_FFMPEG_DIR` environment variable instead of
  a hardcoded path, and fail with a clear message pointing at BtbN LGPL builds.

### Fixed
- Single-pass render no longer drifts against the session audio. Per-segment
  `fps` rounding lost or gained a fraction of a frame at every cut on sources
  that are off the output frame grid (29.97, 23.976, variable frame rate), which
  added up to 0.593 s over 153 cuts on a real 22-minute session. Segments now
  get exact frame counts from cumulative cut boundaries, so the output length
  is `round(total * fps)` regardless of source frame rate.
- Preflight no longer creates an empty directory at the output file path, which
  made the session audio mux fail at 95% with "Is a directory".
- Detection on audio sources no longer collapses on quiet mixer exports: the
  legacy absolute silence floor (0.01, tuned for hot camera audio) gated out
  almost everything (4 segments in 22 minutes on a real session). Sources now use
  a floor of 1e-4.
- Single-pass rendering now uses FFmpeg 7.1+'s file-backed filter option syntax,
  while retaining the legacy filter-script option for older FFmpeg versions.
- GPU preflight now recognizes Apple Silicon acceleration through PyTorch's MPS
  backend instead of incorrectly warning that every Mac has no GPU.

### Removed
- The GPL `imageio-ffmpeg` fallback binary is no longer bundled into the
  installer (licensing; see `docs/THIRD_PARTY.md`).
- `requirements.txt` and `requirements.in` (duplicated and contradicted `pyproject.toml`).
- `PyQt6.sip` hidden import and `pyqt6-sip` dependency (PySide6 uses shiboken6, bundled automatically).
- Committed scratch files (`coverage_report.txt`, `test_new_results.txt`,
  `verification_result.txt`, `verify_settings_persistence.py`, `commands`,
  `shortcuts`); `.gitignore` now blocks them and no longer ignores the
  installer script and app icon that contributors need to build the installer.

> The sections below were reconstructed from git history. These releases were never
> tagged, so dates are approximate (taken from commit dates).

## [1.0.2 – 1.0.3] - 2026-02-08

The version number in the repo jumped from 1.0.1 straight to 1.0.3, so there is no
separate 1.0.2 entry; everything from that period is listed here.

### Added
- New rule-based "fast rules" switching engine, and the switching strategy is now
  selectable in Settings.
- A pre-flight GPU check that warns you before processing starts if your GPU setup
  is not ready, instead of failing in the middle of a render.
- Export of the timeline cuts to FCPXML, so you can keep editing in Final Cut Pro.
- Magic Settings dialog that gathers the AI processing options in one place.
- Audio preview dialog for checking the sound before processing.
- The file list now detects duplicate files and shows a sync status for each one.

### Changed
- The app window is now frameless with a custom title bar.
- The Windows build now bundles the audio-analysis libraries (librosa, soundfile),
  so audio features work out of the box.

### Fixed
- Several fixes in the video preview area of the main window.

## [1.0.1] - 2026-01-14

### Added
- Windows installer (Inno Setup): Zelqivo now installs like a normal Windows
  program, with Start menu and desktop shortcuts, plus a one-step build script.

### Changed
- More detailed log files, which makes problems easier to report and fix.
- The Start menu and desktop shortcuts now use the Zelqivo icon.

### Fixed
- "Fast cut" bug: the automatic edit no longer jumps between cameras too quickly.
  Cuts now respect a minimum clip length and a cooldown between switches, and they
  prefer to land on silent moments.

## [1.0.0] - 2026-01-07

First working build of Zelqivo, shipped as a standalone Windows executable.

### Added
- Multicam podcast editing: drop in the video files from each camera and Zelqivo
  detects who is speaking and cuts to the right camera automatically.
- Active speaker detection with pluggable backends, including an energy-based mode
  that runs on CPU-only machines, and a configurable speaker-to-camera mapping.
- Stage-based processing pipeline with progress tracking and checkpoints, shown in
  a themed progress dialog.
- FFmpeg-based rendering: cutting, effects, an optional QA overlay for reviewing
  the automatic decisions, and merging everything into the final video.
- Settings dialog covering output quality, audio mix, decision engine rules, QA
  overlay, and a light/dark theme toggle.
- Drag-and-drop file list that shows media details for each imported file.
- Standalone Windows executable built with PyInstaller.
