# 2026-05-14 - Autonomous sign policy local validation

## Change

- Added `CONOS` and `VALLA` as autonomous safety-sign classes.
- Safety-sign classes `STOP`, `PROHIBIDO`, `CONOS`, and `VALLA` now command neutral stop for `5.0 s` by default, then ignore safety signs for `5.0 s` by default so the car can continue past the same sign.
- `VELOCIDAD-MAX-30` now uses autonomous throttle `0.50`.
- `VELOCIDAD-MAX-90` now uses autonomous throttle `0.70`.
- EPC remains the control owner; Jetson remains inference-only.

## Local validation

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/py312/bin/python -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/lidar_processor.py
git diff --check
PYTHONPATH=servicios python3 -m unittest discover -s tests -p 'test_autonomous_driver.py'
PYTHONPATH=servicios /opt/homebrew/Caskroom/miniconda/base/envs/py312/bin/python -m unittest discover -s tests -p 'test_coche_runtime.py'
PYTHONPATH=servicios /opt/homebrew/Caskroom/miniconda/base/envs/py312/bin/python -m unittest discover -s tests
```

## Results

- `py_compile`: OK.
- `git diff --check`: OK.
- `test_autonomous_driver.py`: 17 tests OK.
- `test_coche_runtime.py`: 26 tests OK.
- Full local suite: 68 tests OK on current `origin/main`.

## Notes

- The default `python3` in `/tmp` resolved to Python 3.7 without `cv2`; `coche.py` validation used the same Python 3.12 environment used by the main local checkout.
- Physical EPC/car validation still requires deployment and runtime checks on the lab path.
