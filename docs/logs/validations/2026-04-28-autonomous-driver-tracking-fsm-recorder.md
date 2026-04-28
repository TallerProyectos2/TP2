# 2026-04-28 - Autonomous Driver Tracking, FSM, and Recorder

## Scope

Implement a more robust autonomous driving layer for `servicios/coche.py` while preserving the current architecture:

- EPC owns control and web runtime.
- Jetson remains inference-only at `http://100.115.99.8:9001`.
- Car continues receiving only UDP `C + steering + throttle` commands from EPC.

## Changes

- Reworked `servicios/autonomous_driver.py` into a persistent controller:
  - normalized Roboflow detections
  - temporal sign tracker
  - area-based distance proxy without camera recalibration
  - sign selection by class priority, confidence, persistence, zone, and proximity
  - finite-state maneuver controller for cruise, confirming, approach, stop hold, turns, cooldown, ambiguity, and safe fallback
  - command rate limiting for smoother steering/throttle
- Integrated the controller into `servicios/coche.py`.
- Added session/dataset recording:
  - `POST /recording`
  - `GET /recording.json`
  - `recording` section in `/status.json`
  - output directory controlled by `TP2_SESSION_RECORD_DIR`
  - `manifest.jsonl` with frame metadata, predictions, autonomous decision, selected control, backend, and latency
- Updated the web UI with dataset recording controls and recorder status.
- Updated operating docs for the expanded autonomous contract.

## Remote Read-Only State

EPC:

- `git -C /home/tp2/TP2_red4G status --short --branch` -> clean, on `main...origin/main`
- `tp2-car-control.service` -> `inactive`
- no active listener observed on `8088/TCP` or `20001/UDP`

Jetson:

- `tp2-roboflow-inference.service` -> `active`
- `9001/TCP` listening
- EPC -> Jetson `GET http://100.115.99.8:9001/info` -> `Roboflow Inference Server 1.1.2`

No remote service was restarted during this validation.

## Local Validation

Compile:

```bash
/Users/mario/miniconda3/envs/test/bin/python -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/roboflow_runtime.py
```

Result: OK.

Unit tests:

```bash
PYTHONPATH=servicios python3 -m unittest tests/test_autonomous_driver.py
```

Result: `Ran 9 tests ... OK`.

Local runtime smoke test:

```bash
TP2_BIND_IP=127.0.0.1 TP2_BIND_PORT=29001 TP2_WEB_HOST=127.0.0.1 TP2_WEB_PORT=18088 TP2_ENABLE_INFERENCE=0 TP2_SESSION_RECORD_DIR=/tmp/tp2-autonomy-recorder-test /Users/mario/miniconda3/envs/test/bin/python -u servicios/coche.py
```

Observed through HTTP/UDP test:

- fake UDP `I` frames decoded: `video.has_video=true`, `video.frames=2`, `udp.packets={"I": 2}`
- `POST /mode {"mode":"autonomous"}` switched `/status.json` to `control.mode=autonomous`
- with inference disabled, autonomy correctly returned `safe-neutral` / `safe`
- `POST /recording {"action":"start"}` enabled recorder and returned a session path under `/tmp/tp2-autonomy-recorder-test/...`

Recorder write check:

- direct `SessionRecorder` unit smoke wrote `1` image and `1` manifest entry to a temporary directory.

## Limits

This validation did not execute an active LTE end-to-end drive because the EPC control runtime was inactive during the read-only check. Activation on the real lab still requires starting/restarting `coche.py` on EPC in a controlled session and validating car movement behavior.
