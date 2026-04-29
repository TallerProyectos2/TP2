# 2026-04-29 - Ops MQTT idempotence and earlier autonomy decisions

## Scope

Investigate and fix the automated `ops/` startup path after observing the car LED alternating connected/disconnected when the lab is started by systemd automation, while manual startup works.

Changes made:

- `ops/bin/tp2-up` no longer restarts the car-side systemd service by default. The car remains operator-managed unless `TP2_RESTART_CAR_ON_UP=1`.
- `tp2-up` starts Mosquitto with the broker unit (`sudo systemctl start mosquitto`, falling back to `mosquitto.service` for old sudoers compatibility).
- `tp2-up` clears stale retained payloads on `1/command` before startup publish when `TP2_MQTT_CLEAR_RETAINED_ON_UP=1`.
- `tp2-up` publishes `AM-Cloud` once with `mosquitto_pub -q 1` and no retained flag by default.
- The legacy `tp2-car-command-am-cloud.service` remains installed but now publishes non-retained (`-q 1`, no `-r`) and is no longer the normal `tp2-up` publish path.
- Autonomous detection thresholds were tuned to decide earlier:
  - `TP2_INFERENCE_MIN_INTERVAL_SEC=0.07`
  - `TP2_AUTONOMOUS_MIN_AREA_RATIO=0.003`
  - `TP2_AUTONOMOUS_NEAR_AREA_RATIO=0.030`

## Investigation evidence

EPC state before deployment:

- EPC repo was at `0925343` with pre-existing local deletion `D servicios/test.jpg`.
- `mosquitto.service`: active.
- `tp2-car-control.service`: active.
- `tp2-car-command-am-cloud.service`: inactive.
- Broker retained payload on `1/command`: `AM-Cloud`.
- Installed `tp2-car-command-am-cloud.service` on EPC was publishing with `mosquitto_pub -q 1 -r`, so the car command was retained and replayed on every MQTT reconnect.
- Mosquitto logs showed previous start/stop cycles, including a past start-failure loop on `2026-04-27`; current broker was active and listening.

The retained `AM-Cloud` payload is a plausible contributor to the observed LED loop because every client reconnect receives the command again. The old `tp2-up` also restarted the car-side service unconditionally after discovering car SSH, contradicting the docs that say the car runtime is operator-managed.

## Local validation

Commands:

```bash
bash -n ops/lib/tp2-common.sh ops/bin/tp2-up ops/bin/tp2-down ops/bin/tp2-validate ops/bin/tp2-status ops/bin/tp2-install-systemd ops/bin/tp2-install-sudoers
/Users/mario/miniconda3/envs/test/bin/python -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/roboflow_runtime.py
PYTHONPATH=servicios /Users/mario/miniconda3/envs/test/bin/python -m unittest discover -s tests -p 'test_autonomous_driver.py'
PYTHONPATH=servicios /Users/mario/miniconda3/envs/test/bin/python -m unittest discover -s tests -p 'test_coche_runtime.py'
```

Result:

- Shell syntax checks: OK.
- `test_autonomous_driver.py`: 15 tests OK.
- `test_coche_runtime.py`: 10 tests OK.

Local runtime smoke with isolated ports exposed:

- `min_area_ratio=0.003`
- `near_area_ratio=0.03`
- `cruise_throttle=0.65`
- `confirm_frames=1`
- `turn_hold_sec=1.2`
- `turn_degrees=90`

## GitHub

Pushed `main` to GitHub:

- `39e47de` - stabilize MQTT startup and earlier autonomy decisions

## EPC deployment

EPC checkout:

```bash
cd /home/tp2/TP2_red4G
git pull --ff-only origin main
```

Result:

- EPC fast-forwarded to `39e47de`.
- `git status --short --branch` returned `main...origin/main` plus pre-existing `D servicios/test.jpg`.

Installed EPC systemd unit templates:

```bash
TP2_LAB_CONFIG=/tmp/tp2-local-install.env ops/bin/tp2-install-systemd epc
```

Result:

- `tp2-car-command-am-cloud.service` now has:
  - `ExecStart=/bin/sh -c 'exec /usr/bin/mosquitto_pub -q 1 ...'`
  - no retained `-r` flag.

MQTT retained cleanup:

- Before cleanup: retained `1/command` payload was `AM-Cloud`.
- Ran `tp2_prepare_car_mode_topic` locally on EPC with `TP2_EPC_SSH=local`.
- After cleanup: retained `1/command` payload was empty.

Runtime reload:

```bash
sudo -n systemctl stop tp2-car-control.service
sudo -n systemctl start tp2-car-control.service
```

Result:

- `tp2-car-control.service`: active.
- main PID: `83727`.
- worker PID: `83734`.
- UDP listener: `172.16.0.1:20001`.
- web listener: `0.0.0.0:8088`.

Remote validation:

```bash
bash -n ops/lib/tp2-common.sh ops/bin/tp2-up ops/bin/tp2-down ops/bin/tp2-validate ops/bin/tp2-status
python -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/roboflow_runtime.py
PYTHONPATH=servicios python -m unittest discover -s tests -p test_autonomous_driver.py
PYTHONPATH=servicios python -m unittest discover -s tests -p test_coche_runtime.py
```

Result:

- shell syntax checks: OK.
- `test_autonomous_driver.py`: 15 tests OK.
- `test_coche_runtime.py`: 10 tests OK.

Live EPC status after reload:

- `mosquitto`: active.
- `tp2-car-control.service`: active.
- retained `1/command` payload: empty.
- `/status.json` exposed:
  - `autonomy.config.min_area_ratio=0.003`
  - `autonomy.config.near_area_ratio=0.03`
  - `autonomy.config.cruise_throttle=0.65`
  - `autonomy.config.confirm_frames=1`
  - `autonomy.config.turn_hold_sec=1.2`
  - `autonomy.config.turn_degrees=90`
  - `control.effective_steering=0.17`
  - `control.steering_trim=-0.08`
  - `udp.bind=172.16.0.1:20001`

Sudo capability check:

- `sudo -n -l /usr/bin/systemctl start mosquitto`: OK.
- `sudo -n -l /usr/bin/systemctl start mosquitto.service`: OK.

## Runtime note

No firmware changes were made. eNodeB remains radio-only, Jetson remains inference-only, and EPC remains the runtime/control host. Full car LED behavior still needs an observed live startup with the physical car, but the retained MQTT replay and unconditional car-service restart causes have been removed from the automation path.
