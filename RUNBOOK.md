# TP2 Operating Runbook (Script-First)

## Purpose

Define the real startup, validation, and shutdown sequence for the current operational model based on `servicios/`.

## Startup Order

Recommended operator entrypoint:

- From a MacBook with Tailscale access, run `ops/bin/tp2-up jetson` from the repo root.
- The launcher reaches EPC and Jetson directly and reaches the eNodeB through EPC as SSH proxy.

For automated startup, use `ops/bin/tp2-up` as documented in `docs/AUTOSTART.md`.
The manual order below remains the operational source for troubleshooting.

1. Validate EPC <-> eNodeB backhaul connectivity (`10.10.10.1` <-> `10.10.10.2`).
2. Verify eNodeB can reach EPC over the backhaul. `tp2-enb-link.service` runs `/home/tp2/to_epc_link.sh` automatically at eNodeB boot and is not part of the session startup flow.
3. On eNodeB, verify `bladeRF` is connected, confirm `/home/tp2/.config/srsran/enb.conf` uses the intended radio profile, and load `/home/tp2/Descargas/hostedxA9.rbf` with `bladeRF-cli`. The current eNodeB profile is 2x2 MIMO with `tm = 4` and `nof_ports = 2`.
4. Start `srsepc` on EPC.
5. Start `srsenb` on eNodeB.
6. Verify S1 setup is established.
7. Verify car UE attach and current IP assignment from the latest `srsepc` log entry for IMSI `901650000052126`. The old fixed target was `172.16.0.2`, but the live HSS was observed with dynamic allocation on `2026-04-27`. The automated `tp2-up` path does not block on this check by default.
8. Start required services on EPC:
   - `mosquitto`
   - `tp2-car-control.service` (`servicios/coche.py`)
   - inference endpoint (`start_local_inference_server.py`) if local inference path is needed
   - live video/control web view from `coche.py` on `0.0.0.0:8088`
9. Publish the current car mode when required:
   - Normal automation runs `ops/bin/tp2-mqtt-ensure-car-mode`.
   - It keeps `AM-Cloud` retained on `1/command`, verifies the retained value, and skips publishing when that retained state is already present.
   - If another retained payload is found on the same topic, automation logs a conflict before replacing it unless `TP2_MQTT_FAIL_ON_CONFLICT=1`.
10. Open the live operator view from Tailscale at `http://100.97.19.112:8088/`.
    - Keep manual mode selected for initial safety checks.
    - Switch to autonomous mode only after live frames, fresh inference status, lane status, and LiDAR status are visible when the LiDAR is connected.
    - Use the stage toggle to switch between the annotated camera stream and LiDAR reconstruction.
    - Normal systemd sessions autostart dataset recording; stop it from the web UI only when disk space or scene setup makes capture undesirable.
11. If using Jetson offload, first verify Jetson reachability and `tp2-roboflow-inference.service`; then point EPC to `http://100.115.99.8:9001` (or the current reachable Jetson IP) with `TP2_INFERENCE_TARGET=model` and `ROBOFLOW_MODEL_ID=tp2-g4-2026/2`.

## Shutdown Order

1. Stop car activity.
2. Stop EPC services (car control runtime and optional local inference server).
3. Stop `srsenb`.
4. Stop `srsepc`.
5. Keep Jetson inference running unless it was started only for this session and `TP2_STOP_JETSON_ON_DOWN=1`.
6. Collect logs and validation evidence if session changed system state.

## Default Validation Sequence

## LTE Validation

- EPC and eNodeB configs aligned.
- `srsenb` reaches `srsepc`.
- S1 setup completes.
- Car UE attaches and keeps expected IP mapping.

## Script Control Validation

- Selected control script binds its UDP port.
- `coche.py` exposes the live operator web view on `8088/TCP` when used as the control runtime.
- `coche.py` accepts direct remote manual control over the web view and falls back to neutral when web commands stop.
- `coche.py` exposes `POST /mode` for `manual`/`autonomous`; autonomous mode falls back to neutral when frames or inference become stale.
- Autonomous forward movement defaults to positive throttle `+0.65`; reverse throttle is not emitted by the autonomous controller.
- UDP control output applies `TP2_STEERING_TRIM` before sending commands to the car, except during autonomous open turns where trim is bypassed to keep full lock. The default is `-0.24`; `/status.json` reports requested `steering`, `applied_steering_trim`, and sent `effective_steering`.
- The live web UI can change steering compensation (`POST /steering-trim`), autonomous cruise speed (`POST /cruise-speed`), and the optional periodic right-turn pulse (`POST /turn-compensation`) without restarting the service.
- Lane assist is enabled by default with `TP2_LANE_ASSIST_ENABLED=1`; it detects the blue/green tape on the black carpet and applies a bounded correction up to `TP2_LANE_MAX_CORRECTION=0.75` only to autonomous forward actions. It prefers the right corridor when several lanes are visible, slows during strong lane recovery, and exposes `lane.status`, `lane.guidance` and `lane.applied_correction`.
- Autonomous inference cadence defaults to `0.07 s` minimum spacing between submitted frames.
- Autonomous sign selection accepts smaller/farther signs by default (`TP2_AUTONOMOUS_MIN_AREA_RATIO=0.003`, `TP2_AUTONOMOUS_NEAR_AREA_RATIO=0.030`); STOP detections stop immediately and turn actions can begin earlier.
- Turn signs trigger a full-lock open-loop 90-degree maneuver on the first valid confirmed detection, including far detections (`TP2_AUTONOMOUS_TURN_HOLD_SEC`, default `1.20 s`; `TP2_AUTONOMOUS_TURN_DEGREES`, default `90`).
- `coche.py` exposes `POST /recording` and `GET /recording.json` for session capture; recordings include candidate frames, annotated MP4 video, predictions, critical flags, autonomous estimates, and selected controls.
- Normal `tp2-car-control.service` sessions autostart recording under `/srv/tp2/frames/autonomous` and write `manifest.jsonl`, `labels.jsonl`, `critical.jsonl`, `session.mp4`, and optional critical images.
- The live web UI can launch the retraining/replayer server with `POST /replayer/start`; the replayer reads `/srv/tp2/frames/autonomous` directly and provides a session selector.
- Offline review can also run with `python servicios/session_replayer.py /srv/tp2/frames/autonomous` and writes `labels_reviewed.json` without modifying the original manifest.
- Script receives car payloads (`I`, `B`, `D`).
- Script receives LiDAR payloads (`L`) or LiDAR nested in telemetry `D` when the sensor is connected.
- Script sends control packets (`C`) back to car.
- Car behavior matches command stream.

## LiDAR Validation

- Confirm `coche.py` receives `L` packets or telemetry `D` with `ranges`/`points`.
- Confirm `/status.json` reports `lidar.frames`, `lidar.status`, `lidar.safety.min_front_distance_m`, and a bounded `lidar.points` list.
- Confirm the web stage can switch from camera to LiDAR reconstruction.
- In autonomous mode, place an obstacle inside `TP2_LIDAR_STOP_DISTANCE_M` and confirm the selected action becomes `lidar-stop` with neutral throttle.
- Move the obstacle into the slow band between `TP2_LIDAR_STOP_DISTANCE_M` and `TP2_LIDAR_SLOW_DISTANCE_M` and confirm speed limiting without moving control off EPC.

## EPC Inference Validation

- Local inference endpoint reachable when enabled (default `127.0.0.1:9001`).
- `inferencia.py` runs with a known image and writes annotated output.
- `coche.py` reports the configured inference backend on `/status.json`.
- Live `coche.py` inference passes OpenCV NumPy frames to `inference_sdk` directly; it does not create a temporary JPEG per inference request.

## Jetson Validation (when enabled)

- EPC can reach Jetson inference endpoint (`100.115.99.8:9001` in the last validated lab state).
- Script configuration can switch to Jetson model target without moving UDP control off EPC.
- `inferencia.py` on EPC completes successfully against the Jetson endpoint.
- Fallback to EPC local inference is validated.

## Operational Rules

- Prefer read-only inspection first.
- Avoid restarting healthy services unless required by the task.
- Do not change more than one layer at once during troubleshooting.
- If LTE is unstable, stop before script-level or Jetson-level debugging.
- Firmware updates are forbidden on all components.

## Troubleshooting Order

1. EPC and eNodeB process state
2. Backhaul network and S1 state
3. UE attach and IP assignment
4. UDP control script RX/TX path
5. EPC local inference endpoint
6. Jetson inference path (only if enabled)
7. Car-side execution path

## Escalation Conditions

Stop and escalate if:

- credentials are missing,
- remote state is ambiguous,
- a destructive action is required and impact is unclear,
- validation cannot be completed,
- documentation and live system behavior diverge in a way that cannot be explained.
