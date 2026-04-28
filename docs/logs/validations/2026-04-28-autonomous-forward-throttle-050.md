# 2026-04-28 - Autonomous forward throttle +0.50

## Scope

Set autonomous forward movement to a positive throttle of `+0.50` and prevent autonomous reverse throttle.

## Changes validated locally

- Autonomous forward throttle defaults:
  - `crawl_throttle=0.50`
  - `slow_throttle=0.50`
  - `turn_throttle=0.50`
  - `cruise_throttle=0.50`
  - `fast_throttle=0.50`
- The autonomous command filter clamps throttle to `[0.0, 1.0]`, so negative/reverse throttle cannot be emitted by autonomous decisions.
- Safe fallback, stop, ambiguity and stale data remain neutral.

## Commands

- `python -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/roboflow_runtime.py`
- `PYTHONPATH=servicios python -m unittest tests/test_autonomous_driver.py tests/test_coche_runtime.py`
  - result: 16 tests passed
- Local `coche.py` smoke with `TP2_ENABLE_INFERENCE=0` confirmed `/status.json` exposes all autonomous throttle defaults as `0.5`.

## Note

The local smoke keeps throttle at `0.0` because inference/video are disabled, which is the expected safe fallback. Unit tests validate active autonomous forward decisions use raw throttle `0.50`.
