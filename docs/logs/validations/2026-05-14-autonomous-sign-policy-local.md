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

## GitHub deployment

- Commit pushed to GitHub `main`: `7bfa7b9` (`Update autonomous sign speed and stop policy`).
- GitHub repository redirect observed: `TallerProyectos2/TP2_red4G` now redirects to `TallerProyectos2/TP2`.

## EPC deployment

Copied the updated runtime/docs/test files to `/home/tp2/TP2_red4G` on `tp2-EPC` with `rsync`.

The EPC host-local defaults file `/home/tp2/.config/tp2/coche-control-defaults.json` was preserving older runtime values, so it was updated in place after creating backup `/home/tp2/.config/tp2/coche-control-defaults.json.bak-20260514`:

```json
{"fast_throttle": 0.7, "slow_throttle": 0.5, "stop_hold_sec": 5.0, "stop_ignore_sec": 5.0}
```

Remote validation on EPC:

```bash
/home/tp2/miniforge3/bin/conda run --no-capture-output -n tp2 python -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/lidar_processor.py
PYTHONPATH=servicios /home/tp2/miniforge3/bin/conda run --no-capture-output -n tp2 python -m unittest discover -s tests -p 'test_autonomous_driver.py'
PYTHONPATH=servicios /home/tp2/miniforge3/bin/conda run --no-capture-output -n tp2 python -m unittest discover -s tests -p 'test_coche_runtime.py'
PYTHONPATH=servicios /home/tp2/miniforge3/bin/conda run --no-capture-output -n tp2 python -m unittest discover -s tests
git diff --check
```

Remote results:

- `py_compile`: OK.
- `git diff --check`: OK.
- `test_autonomous_driver.py`: 17 tests OK.
- `test_coche_runtime.py`: 26 tests OK.
- Full EPC suite: 70 tests OK.

Runtime activation:

```bash
sudo -n systemctl stop tp2-car-control.service
sudo -n systemctl start tp2-car-control.service
```

Final live EPC status from `http://127.0.0.1:8088/status.json`:

```json
{
  "control_mode": "manual",
  "slow_throttle": 0.5,
  "fast_throttle": 0.7,
  "stop_hold_sec": 5.0,
  "stop_ignore_sec": 5.0,
  "udp_packets": {"B": 12, "D": 222, "I": 695, "L": 173},
  "udp_tx_packets": 1561,
  "inference_status": "running",
  "lidar_status": "clear",
  "imu_status": "ready"
}
```

Physical validation still remains: drive the car in autonomous mode past `STOP`, `PROHIBIDO`, `CONOS`, and `VALLA` scenes and confirm the observed 5 s stop followed by continuation.
