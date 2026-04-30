# 2026-04-30 - Live tuning and retraining UI validation

## Scope

- Added runtime control tuning to `servicios/coche.py` for manual throttle, autonomous cruise/turn pulse values, steering trim, and lane-assist correction.
- Added host-local persistence for current control defaults through `POST /settings/defaults`.
- Improved `servicios/session_replayer.py` with MP4 playback, timeline/jump navigation, faster relabel controls, session metadata editing, safe session rename, and manifest-safe frame image rename.

## Local validation

Commands run from repo root:

```bash
python -m py_compile servicios/coche.py servicios/roboflow_runtime.py servicios/session_replayer.py servicios/autonomous_driver.py servicios/lane_detector.py
PYTHONPATH=servicios python -m unittest discover -s tests
node --check /tmp/tp2-live.js
node --check /tmp/tp2-replayer.js
```

Result:

- `44` tests passed.
- No Python compile errors.
- Live and replayer inline JavaScript passed `node --check`.

Runtime smoke test:

- Started `servicios/coche.py` locally with:
  - `TP2_BIND_IP=127.0.0.1`
  - `TP2_BIND_PORT=23001`
  - `TP2_WEB_PORT=18088`
  - `TP2_ENABLE_INFERENCE=0`
  - `TP2_SESSION_RECORD_DIR=/tmp/tp2-codex-record`
  - `TP2_CONTROL_DEFAULTS_PATH=/tmp/tp2-codex-defaults.json`
- Sent one synthetic `I + pickle(jpeg)` UDP frame.
- Confirmed response packet type `C`.
- Confirmed `GET /status.json` reported:
  - `video.has_video=true`
  - `video.frames=1`
  - `udp.packets.I=1`
  - `settings.values` present
- Confirmed `POST /settings` updated `cruise_throttle=0.42`, `steering_trim=-0.30`, and `turn_pulse_enabled=false`.
- Confirmed `POST /settings/defaults` wrote `/tmp/tp2-codex-defaults.json`.
- Confirmed `POST /replayer/start` opened the replayer on `18090`.
- Confirmed `GET /api/sessions` listed the recorded session.
- Confirmed `POST /api/session/meta` wrote review metadata.
- Confirmed `POST /api/frame/rename` renamed the selected frame image and updated `manifest.jsonl`.
- Confirmed `GET /video.mp4` with `Range: bytes=0-31` returned `32` bytes.

## EPC pre-deploy check

Read-only check before deployment:

- SSH to `tp2@100.97.19.112` succeeded with key-based auth.
- Hostname: `tp2-EPC`.
- `/home/tp2/TP2_red4G` was on `main...origin/main`.
- Existing unrelated EPC worktree state: `D servicios/test.jpg`.
- `tp2-car-control.service` was active.
- `8088/TCP` and `8090/TCP` were listening from the current runtime process.
