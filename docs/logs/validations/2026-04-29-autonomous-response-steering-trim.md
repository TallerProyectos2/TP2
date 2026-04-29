# 2026-04-29 - Autonomous response tuning and steering trim

## Scope

Tune the EPC autonomous driving runtime for the live car:

- autonomous forward throttle defaults changed from `+0.50` to `+0.65`
- inference submit interval default reduced from `0.18 s` to `0.10 s`
- autonomous sign confirmation default reduced to one valid frame
- turn-sign maneuvers marked/configured as 90-degree open-loop turns with `1.20 s` default hold
- outgoing UDP steering now applies `TP2_STEERING_TRIM=-0.08` by default to compensate the physical left drift; status exposes both requested `steering` and sent `effective_steering`

## Local validation

Commands:

```bash
/Users/mario/miniconda3/envs/test/bin/python -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/roboflow_runtime.py
PYTHONPATH=servicios /Users/mario/miniconda3/envs/test/bin/python -m unittest discover -s tests -p 'test_autonomous_driver.py'
PYTHONPATH=servicios /Users/mario/miniconda3/envs/test/bin/python -m unittest discover -s tests -p 'test_coche_runtime.py'
```

Result:

- `test_autonomous_driver.py`: 13 tests OK
- `test_coche_runtime.py`: 10 tests OK

Isolated local smoke:

- `coche.py` started on `127.0.0.1:29066` and web `127.0.0.1:18066`
- `/status.json` exposed:
  - `steering=0.25`
  - `effective_steering=0.17`
  - `steering_trim=-0.08`
  - `cruise_throttle=0.65`
  - `confirm_frames=1`
  - `turn_hold_sec=1.2`
  - `turn_degrees=90`

## GitHub

Pushed `main` to GitHub:

- `0d943b7` - autonomous response tuning
- `89f5883` - steering trim compensation

## EPC deployment

EPC checkout:

```bash
cd /home/tp2/TP2_red4G
git pull --ff-only origin main
```

Result:

- EPC fast-forwarded to `89f5883`
- `git status --short --branch` returned `main...origin/main` plus the pre-existing local deletion `D servicios/test.jpg`

Remote EPC validation commands:

```bash
python -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/roboflow_runtime.py
PYTHONPATH=servicios python -m unittest discover -s tests -p test_autonomous_driver.py
PYTHONPATH=servicios python -m unittest discover -s tests -p test_coche_runtime.py
```

Result:

- `test_autonomous_driver.py`: 13 tests OK
- `test_coche_runtime.py`: 10 tests OK

Runtime reload:

- `sudo -n systemctl restart tp2-car-control.service` was unavailable because sudo requested a password.
- The old user-owned Python child was killed so `Restart=on-failure` relaunched the active systemd service with the new checkout.
- Main PID changed from `70359` to `72463`; worker process became `72468`.

Live EPC status after reload:

- `tp2-car-control.service`: active
- `tp2-srsepc.service`: active
- `mosquitto.service`: active
- `srs_spgw_sgi`: `172.16.0.1/24`
- UDP listener: `172.16.0.1:20001`
- web listener: `0.0.0.0:8088`
- Jetson inference endpoint from EPC: `http://100.115.99.8:9001/info` returned `Roboflow Inference Server 1.1.2`
- `/status.json` exposed:
  - `control.mode=manual`
  - `control.source=neutral`
  - `control.steering=0.25`
  - `control.effective_steering=0.17`
  - `control.steering_trim=-0.08`
  - `control.throttle=0.0`
  - `autonomy.config.cruise_throttle=0.65`
  - `autonomy.config.turn_throttle=0.65`
  - `autonomy.config.confirm_frames=1`
  - `autonomy.config.turn_hold_sec=1.2`
  - `autonomy.config.turn_degrees=90`
  - `udp.bind=172.16.0.1:20001`

## Runtime note

No firmware changes were made. eNodeB remains radio-only, Jetson remains inference-only, and EPC remains the runtime/control host.
