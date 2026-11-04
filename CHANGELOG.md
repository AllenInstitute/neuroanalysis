# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases before 0.0.8 predate this file; see the commit history for those.

## [Unreleased]

## [0.0.8] - 2026-11-04

### Added

- `PatchClampTestPulse.label_text()` returns the HTML analysis summary that
  `label_for_plot()` renders, so callers can display it outside a pyqtgraph plot.
- Stimulus constructors validate their arguments instead of silently building
  nonsense: `SquarePulse`, `Ramp` and `Sine` reject negative durations, `Sine`
  rejects non-positive frequencies, and `SquarePulseTrain` rejects non-integer or
  non-positive `n_pulses` and intervals shorter than the pulse duration.

### Changed

- `SquarePulseSeries` raises `ValueError` rather than tripping an `assert` when
  `pulse_times`, `pulse_durations` and `amplitudes` differ in length, so the
  check survives `python -O`.
- `requires-python` declares the actual floor of 3.10.
- `neuron` is no longer a base dependency; it is available through the
  `neuron_sim` and `test` extras. The `test` extra also declares `lmfit`.

### Fixed

- `H5BackedTestPulseStack.at_time()` raised `AttributeError` on every call by
  guarding on a `_readable` attribute the class never sets. The guard is gone.
- Fitting a PSP no longer requires pyqtgraph. `fit_psp()` falls back to a no-op
  profiler when pyqtgraph — or a Qt binding for it — is unavailable.
- Test pulse capacitance is reported as NaN instead of raising
  `ZeroDivisionError` when input or access resistance is non-positive.
- NumPy 2 compatibility: event detection uses `scipy.integrate.trapezoid` in
  place of the removed `np.trapz`, and `searchfit` uses `np.prod` in place of
  the removed `np.product`.

### Development

- CI runs the test suite on Python 3.10 through 3.14, on every push and pull
  request.
- The PSP fitting tests are skipped and the test pulse noise cases are xfailed
  pending regenerated baselines.

[Unreleased]: https://github.com/AllenInstitute/neuroanalysis/compare/0.0.8...HEAD
[0.0.8]: https://github.com/AllenInstitute/neuroanalysis/releases/tag/0.0.8
