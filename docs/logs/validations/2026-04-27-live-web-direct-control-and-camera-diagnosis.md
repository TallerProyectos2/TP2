# 2026-04-27 - Live web direct control and camera diagnosis

## Scope

- Removed the visible arm/neutral controls from the live web UI.
- Changed `servicios/coche.py` so `POST /control` applies web steering/throttle directly when web control is enabled.
- Kept `/control/neutral` and the watchdog path for safe fallback when the browser stops publishing commands.
- Investigated why live camera frames and Jetson inference were not visible.

## Local validation

- Syntax check:
  - `python3 -m py_compile servicios/coche.py`
  - result: OK.
- Runtime check on loopback:
  - launched `coche.py` with `TP2_BIND_IP=127.0.0.1`, `TP2_BIND_PORT=29001`, `TP2_WEB_PORT=18088`, and inference disabled.
  - sent a synthetic JPEG frame as `I + pickle(jpeg)` over UDP.
  - observed UDP reply `C` with neutral steering/throttle.
  - `GET /status.json` reported `video.has_video=true`, `video.frames=2`, and packets `{"I": 2}`.
  - `GET /snapshot.jpg` returned JPEG bytes starting with `ffd8ffe0`.
  - `POST /control` with `{"steering": -0.75, "throttle": 0.5}` returned `armed=true`, `source=web`, `steering=-0.75`, `throttle=0.5` without requiring an `armed` field.

## Live read-only checks

- `ops/bin/tp2-status`:
  - EPC `srsepc`: active.
  - eNodeB `srsenb`: active.
  - S1 association: established.
  - EPC UDP listener: `172.16.0.1:20001`.
  - EPC web listener: `0.0.0.0:8088`.
  - Jetson inference service: active.
  - Jetson OpenAPI: reachable.
- Jetson direct check:
  - `http://100.115.99.8:9001/info` returned Roboflow Inference Server `1.1.2`.
- Live web status before code deployment:
  - UDP client: `172.16.0.4:42141`.
  - packets: only `B` packets observed.
  - `video.has_video=false`, `video.frames=0`, `video.decode_errors=0`.
  - inference status: `waiting-frame`.
- EPC `srsepc` log:
  - IMSI `901650000052126` attached as `172.16.0.4` in the current session.
- EPC HSS file:
  - `/home/tp2/.config/srsran/user_db.csv` currently has `IP_alloc=dynamic` for IMSI `901650000052126`.
  - This diverges from the older documented fixed `172.16.0.2` assignment and explains why the live UDP client is `172.16.0.4`.
- Professor reference scripts:
  - `scripts_profesor/car1_manual_control_server.py` and `scripts_profesor/car1_cloud_control_server_real_time_control.py` both parse UDP as `data_type = data[0]`, `pickle.loads(data[1:])`.
  - Camera frames are decoded only for `data_type == b"I"` with `cv2.imdecode(...)`.

## Diagnosis

The EPC web runtime and Jetson endpoint are healthy. The missing detections are caused by the absence of camera image packets from the car: the active runtime receives battery packets (`B`) but no image packets (`I`). Because no frame reaches `coche.py`, the inference worker correctly remains in `waiting-frame`.

Republishing the documented MQTT mode `AM-Cloud` to `1/command` did not change the packet mix during the observation window: the EPC continued receiving only `B` packets.

## Blocker

Car-side SSH from EPC to `172.16.0.4` is reachable but non-interactive login failed for known users (`tp2`, `artemis`, `pi`, `ubuntu`, `grupo4`, `root`) due to missing credentials. Without car-side credentials, this task cannot inspect or restart the camera capture process that should emit `I` packets.

## Deployment

- Deployed the updated `servicios/coche.py` to `/home/tp2/TP2_red4G/servicios/coche.py` on EPC.
- Updated and deployed `ops/bin/tp2-status` so it reports the latest observed car UE IP for IMSI `901650000052126` instead of treating stale `172.16.0.2` log lines as current evidence.
- Remote syntax check:
  - `PYTHONNOUSERSITE=1 /home/tp2/miniforge3/envs/tp2/bin/python -m py_compile servicios/coche.py`
  - result: OK.
  - `bash -n ops/bin/tp2-status ops/lib/tp2-common.sh`
  - result: OK.
- Restarted `tp2-car-control.service` with the narrow passwordless systemd permission.
- Final runtime state:
  - `tp2-car-control.service`: active.
  - UDP listener: `172.16.0.1:20001` owned by the service Python process.
  - Web listener: `0.0.0.0:8088` owned by the service Python process.
  - Web HTML no longer contains the visible `ARMAR` or `NEUTRO` controls.
  - `POST /control` without an `armed` field returned `armed=true`, `source=web`.
  - `ops/bin/tp2-status` from the MacBook reports `car UE: latest 172.16.0.4 (configured 172.16.0.2)`.

## Professor-style helper

- Added `scripts_profesor/car1_grupo4.py`.
- This helper is intentionally a near-copy of `scripts_profesor/car1_manual_control_server.py`.
- It binds `172.16.0.1:20001`, receives `I + pickle(jpeg)` from the car, displays frames with OpenCV, and replies with `C + steering + throttle` like the professor script.
- Differences from the professor script are limited to:
  - `server_address = ('172.16.0.1', 20001)`.
  - initial neutral steering `control_giro=0.25`.
  - `recvfrom(131072)` to match the current runtime receive size in `servicios/coche.py`.
- It keeps the professor key mapping:
  - `w` / `s` / `x` / `2` for throttle.
  - `a` / `d` for steering.
- Local validation:
  - `python3 -m py_compile scripts_profesor/car1_grupo4.py`: OK.
- EPC validation:
  - deployed to `/home/tp2/TP2_red4G/scripts_profesor/car1_grupo4.py`.
  - `PYTHONNOUSERSITE=1 /home/tp2/miniforge3/envs/tp2/bin/python -m py_compile scripts_profesor/car1_grupo4.py`: OK.
  - not launched while `tp2-car-control.service` was active because both bind `172.16.0.1:20001`.
