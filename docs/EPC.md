# EPC Machine Specification

## Scope

Machine-specific source of truth for the EPC host under the current script-first operating model.

## Baseline Identity

- Hostname: `tp2-EPC`
- Validation baseline updated: `2026-03-10`

## Network Bindings

- Upstream interface:
  - `eno1`
  - `10.0.128.174/24` (DHCP at last validation)
- eNodeB backhaul:
  - `enp1s0`
  - `10.10.10.1/24`
- SGi side:
  - `srs_spgw_sgi`
  - `172.16.0.1/24`
- Tailscale:
  - `tailscale0`
  - `100.97.19.112/32`

## LTE Core Contract

- Runtime process: `srsepc`
- Primary config: `/home/tp2/.config/srsran/epc.conf`
- HSS DB: `/home/tp2/.config/srsran/user_db.csv`
- Core listeners:
  - `10.10.10.1:36412/SCTP` (S1-MME)
  - `10.10.10.1:2152/UDP` (GTP-U)

## UE Provisioning Contract

- UE pool network: `172.16.0.0/24`
- Previous fixed car mapping target:
  - IMSI `901650000052126` -> `172.16.0.2`
- Live assignment note (`2026-04-27`):
  - `/home/tp2/.config/srsran/user_db.csv` currently shows `IP_alloc=dynamic`
  - latest observed assignment in `srsepc` log: `IMSI: 901650000052126, UE IP: 172.16.0.4`

## Routing And DNS Contract

- `net.ipv4.ip_forward=1`
- Persistent NAT/bootstrap script:
  - `/usr/local/sbin/tp2-ue-routing.sh`
- Startup unit:
  - `tp2-ue-routing.service`
- NAT rule:
  - `POSTROUTING -s 172.16.0.0/24 -o eno1 -j MASQUERADE`
- Optional UE DNS:
  - `dnsmasq` on `172.16.0.1:53`

## Script Runtime Contract On EPC

Operational files are stored in repo `servicios/` and validated on EPC under `/home/tp2/TP2_red4G/servicios/`.

- Car web/control runtime:
  - `coche.py` (`172.16.0.1:20001` UDP, `0.0.0.0:8088` web)
  - `autonomous_driver.py` deterministic autonomous controller used by `coche.py`
  - `lane_detector.py` OpenCV lane detector for the blue/green tape corridors; used by `coche.py` as bounded steering stabilization in autonomous forward actions
  - session recorder under `TP2_SESSION_RECORD_DIR` for dataset candidates, annotated MP4, critical flags, and offline relabeling inputs
  - `session_replayer.py` review server for frame replay and label correction, launchable from the `coche.py` web UI and backed by a session selector over the recording root
  - manual web control is gated by drive mode; stale manual posts cannot switch an active autonomous session back to manual
  - live web tuning can adjust manual throttle, autonomous cruise/turn pulse values, steering trim, lane-assist correction, and turn compensation at runtime through `/settings`; `POST /settings/defaults` persists the current values to a host-local `TP2_CONTROL_DEFAULTS_PATH` file outside the repository
  - autonomous forward commands are clamped to non-negative throttle and default to `+0.65`; the web UI can update the live cruise throttle with `POST /cruise-speed`
  - outgoing UDP steering applies live steering trim (`TP2_STEERING_TRIM=-0.24` default) to compensate the current physical left drift; trim is bypassed during autonomous open turns to preserve full-lock steering; `/status.json` exposes `effective_steering`, and the web UI can update trim with `POST /steering-trim`
  - optional periodic right-turn compensation (`POST /turn-compensation`) can add a short right steering pulse during autonomous forward actions without competing with open turn maneuvers
  - lane assist defaults to enabled (`TP2_LANE_ASSIST_ENABLED=1`) and exposes `lane.status`, `lane.guidance`, and `lane.applied_correction` in `/status.json`; it accepts partially visible edge corridors, prefers the right corridor when several lanes are visible, and can apply up to `TP2_LANE_MAX_CORRECTION=0.75` while slowing during strong recovery.
  - autonomous inference submits frames every `0.07 s` by default when frames are available
  - autonomous sign filtering accepts smaller/farther signs by default (`TP2_AUTONOMOUS_MIN_AREA_RATIO=0.003`, `TP2_AUTONOMOUS_NEAR_AREA_RATIO=0.030`); STOP detections stop immediately and turn decisions begin earlier
  - turn signs trigger a full-lock timed 90-degree open-loop maneuver from the first valid confirmed detection, including far detections
- Inference files:
  - `start_local_inference_server.py` (optional EPC local endpoint, default `127.0.0.1:9001`)
  - `inferencia.py` (CLI execution and annotated output)
  - `roboflow_runtime.py` (shared Roboflow client/helpers for CLI and live car control)
  - live `coche.py` inference uses OpenCV NumPy frames through `inference_sdk`, avoiding temporary JPEG files on disk
  - current remote inference profile for `coche.py`: `/home/tp2/.config/tp2/coche-jetson.env` on EPC
  - current Jetson Roboflow target: direct model inference with `ROBOFLOW_MODEL_ID=tp2-g4-2026/2`
- Conda runtime:
  - env name: `tp2`
  - activation hook: `/home/tp2/miniforge3/envs/tp2/etc/conda/activate.d/tp2-runtime.sh`
  - shell init: `conda init` applied for `zsh` and `bash`
  - operator command: `conda activate tp2`

## Service Port Contract (Current)

- `36412/SCTP`: `srsepc` S1-MME
- `2152/UDP`: `srsepc` GTP-U
- `53/TCP,UDP`: optional `dnsmasq`
- `20001/UDP`: car control runtime (`coche.py`)
- `8088/TCP`: live `coche.py` operator web view with camera, browser control, and inference status
  - `POST /mode`: switch between `manual` and `autonomous`
  - `POST /recording`: start/stop dataset capture
  - `POST /steering-trim`: update live steering compensation before UDP send
  - `POST /cruise-speed`: update live autonomous cruise throttle
  - `POST /turn-compensation`: update optional periodic right-turn pulse settings
  - `GET /recording.json`: inspect current recording state
  - `POST /replayer/start`: start the session replayer on `TP2_SESSION_REPLAYER_PORT` (default `8090`)
  - `POST /settings`: update runtime control tuning
  - `POST /settings/defaults`: persist current runtime tuning as the machine default outside the repo
- `9001/TCP`: local inference endpoint (when started)
- Remote inference offload last validated from EPC to Jetson: `100.115.99.8:9001`

## Storage And Log Paths

- `/srv/tp2/frames`
  - default base for autonomous dataset captures: `/srv/tp2/frames/autonomous`
  - session outputs include `manifest.jsonl`, `labels.jsonl`, `critical.jsonl`, `session.mp4`, raw frames, and reviewed labels from `session_replayer.py`
- `/srv/tp2/logs`
- `/srv/tp2/docker`
- `/srv/tp2/config`
- LTE logs:
  - `/srv/tp2/logs/srsepc.log`
  - `/srv/tp2/logs/srsepc-console.log` (when launched with console redirect)

## Evidence References

- EPC phase baselines:
  - `docs/logs/validations/2026-03-04-epc-fase-0.md`
  - `docs/logs/validations/2026-03-04-epc-fase-1.md`
  - `docs/logs/validations/2026-03-05-epc-fase-2.md`
- EPC local inference:
  - `docs/logs/validations/2026-03-05-epc-inferencia-local.md`
- Car UE assignment:
  - `docs/logs/validations/2026-03-10-car-ue-ip-assignment.md`
