# Car Runtime Contract (Current)

## Purpose

Document the current car-side interaction model used by existing EPC scripts.

## Control Model

The active operational model is UDP stream + UDP control:

- Car sends payloads to EPC control server.
- EPC script processes payloads and returns control command.

No new backend API is required to run this path.

## Runtime Endpoint In Use

Normal sessions use one EPC runtime from `servicios/`:

- `coche.py`

## UDP Packet Contract (As Implemented)

- Incoming payload discriminator (first byte):
  - `I`: camera image payload
  - `B`: battery level
  - `D`: reserved/other data path
  - `L`: LiDAR scan payload
- Payload body is deserialized with `pickle.loads(...)` in current scripts.
- `L` can be sent as `pickle` or raw JSON. The EPC accepts the professor/Artemis plain pickled range list, LaserScan-like `ranges` with `angle_min`/`angle_increment`, or Cartesian `points` as `[x, y, z, intensity]`/objects. LiDAR can also be nested in `D` under `lidar`, `lidar_scan`, `scan`, `ranges`, or `points`.
- Outgoing control packet type:
  - `C` + steering (`double`) + throttle (`double`)

## Runtime Modes

- Web manual mode:
  - direct browser-driven steering/throttle through `coche.py` on EPC
  - camera display for operator feedback
  - optional Roboflow inference overlay on live camera frames
- Web autonomous mode:
  - operator toggles manual/autonomous from the same web UI
  - runtime tuning in the same web UI can adjust manual throttle, autonomous cruise/turn pulse values, steering trim, lane-assist correction, LiDAR safety thresholds/sector, and autonomous timing/detection parameters without moving control off EPC
  - `POST /settings/defaults` saves the currently tuned values to a host-local defaults file; this file is machine configuration and must not be committed
  - EPC uses fresh Roboflow detections to choose continue, turn, stop, crawl, slow, or faster cruise
  - autonomous forward movement uses positive throttle `+0.65`; stop, ambiguity, stale frame or stale inference still force neutral `0.0`; the EPC web UI can change cruise throttle live
  - outgoing UDP steering is trimmed before packet send; current default `TP2_STEERING_TRIM=-0.24` compensates the physical left drift with a stronger rightward correction, the EPC web UI can change that trim live, and open turn maneuvers bypass trim to keep full lock
  - an optional periodic right-turn pulse can compensate left drift during autonomous forward actions without applying during open turn maneuvers
  - `coche.py` also runs OpenCV lane assist on the camera frame: it detects the continuous blue/green tape lines on the carpet, estimates the current corridor, prefers the right corridor when several lanes are visible, slows during strong recovery, and applies a bounded steering correction only during autonomous forward actions
  - `coche.py` also consumes fresh LiDAR scans when connected: the default collision sector is the 45 frontal beams centered on the car direction, obstacles at or below `0.15 m` force `lidar-stop`, nearer slow-zone obstacles cap throttle and add a bounded avoidance steering correction, and stale/missing LiDAR does not move orchestration away from EPC
  - nearest/relevant signs are selected by bounding-box area, confidence, persistence, image zone (`left`, `center`, `right`), and maneuver state
  - default sign thresholds are tuned to act on slightly smaller/farther signs; STOP detections stop immediately and turn decisions begin before the car reaches the sign
  - detections are tracked across frames; default turn decisions trigger full-lock 90-degree maneuvers on the first valid confirmed frame, including far turn detections
  - the FSM holds stops, maintains 90-degree open-loop turns for the configured maneuver window, and applies cooldowns to avoid repeating the same sign
  - stale frame or stale inference state forces neutral instead of continuing on old detections
- Dataset recording mode:
  - web/API controlled session recorder saves image candidates and `manifest.jsonl`
  - normal EPC systemd sessions autostart recording unless disabled by host-local environment
  - records include Roboflow predictions, annotated MP4 video, candidate labels, critical flags, autonomous estimate, selected target, command, backend and latency
  - critical flags include low-confidence detections, class changes on the same recorder track, short-lived detections, ambiguous autonomous decisions, and operator overrides during autonomous mode
  - saved labels are estimates/candidates, not ground truth; they must be curated before reuploading to Roboflow
  - the live web UI can start the retraining/replayer server; `servicios/session_replayer.py` reads the recording root directly, shows a session selector, and writes reviewed labels beside the selected capture

## LTE Binding Context

- Car attaches as UE in EPC network.
- Previous static mapping target:
  - IMSI `901650000052126` -> `172.16.0.2`
- Live note (`2026-04-27`):
  - EPC HSS currently has `IP_alloc=dynamic`; latest observed UE IP was `172.16.0.4`

## Minimum Validation

1. Start LTE (`srsepc` + `srsenb`) and verify UE attach.
2. Start `tp2-car-control.service` on EPC.
3. Confirm script receives UDP payloads from car.
4. If LiDAR is connected, confirm `L` packets or LiDAR telemetry update `/status.json`.
5. Confirm control packets are returned and car responds.

## EPC Operator Notes

- Current LTE-side runtime bind is `172.16.0.1:20001`.
- `coche.py` is the only normal car runtime in `servicios/`.
- `coche.py` exposes the live camera/inference/operator status and remote manual control web view on `8088/TCP`; use `http://100.97.19.112:8088/` from Tailscale during normal EPC-run sessions.
- Browser control is direct only in manual mode once the operator opens the web UI. Neutral manual posts do not arm the control path, and the watchdog returns active manual commands to neutral instead of holding the last throttle. Manual control release does not leave autonomous mode; the Stop button is the explicit manual neutral stop.
- The web UI exposes `POST /mode` for `manual` and `autonomous`. Manual is the safe default; autonomous should only be enabled after live camera frames, inference, and lane status are visible.
- The web UI exposes `POST /steering-trim`, `POST /cruise-speed`, and `POST /turn-compensation` for live steering trim, cruise throttle, and periodic right-pulse tuning.
- The web UI exposes `POST /recording` for dataset capture. Default recording state is controlled by `TP2_SESSION_RECORD_AUTOSTART`; normal systemd operation records sessions by default for retraining evidence.
- The web UI main stage can switch between the annotated camera stream and a top-down LiDAR reconstruction fed by `/status.json`; the LiDAR view shows the car as a direction arrow, surrounding scan points, distance rings, and close-point distance labels.
- `scripts_profesor/car1_grupo4.py` is a professor-style manual-control server adapted for Grupo 4. It intentionally keeps the professor script behavior and binds the LTE runtime address `172.16.0.1:20001`, so do not run it at the same time as `tp2-car-control.service`.
- `coche.py` defaults its live inference endpoint to Jetson at `http://100.115.99.8:9001` using direct model inference (`TP2_INFERENCE_TARGET=model`, `ROBOFLOW_MODEL_ID=tp2-g4-2026/2`); override these variables only when intentionally using another backend.
- `coche.py` loads `/home/tp2/.config/tp2/inference.env` or `/home/tp2/.config/tp2/coche-jetson.env` automatically, so operators do not need to `source` the token manually.
- On the EPC, `conda activate tp2` also loads the same runtime variables and token through the Conda activation hook.
- The validated Jetson offload path keeps EPC as control owner and points inference to `ROBOFLOW_LOCAL_API_URL=http://100.115.99.8:9001` when Jetson is reachable.
- Secrets required for Roboflow access must stay in host-local env files or shell state, never in the repository.
