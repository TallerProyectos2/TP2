# 2026-04-30 - EPC MQTT, cruise speed and turn compensation

## Scope

- MQTT retained `AM-Cloud` helper installed on EPC systemd path.
- `coche.py` turn signs now start full-lock 90-degree maneuvers from the first confirmed detection.
- Web UI/API now exposes live steering trim, cruise throttle, and optional right-pulse compensation.

## Local validation

- `python -m py_compile servicios/autonomous_driver.py servicios/coche.py`: ok
- `PYTHONPATH=servicios python -m unittest discover -s tests`: 49 tests ok
- `git diff --check`: ok
- local `coche.py` HTTP smoke on `127.0.0.1:18088`: `POST /steering-trim`, `POST /cruise-speed`, `POST /turn-compensation`, and `/status.json` ok

## EPC deployment

- EPC checkout fast-forwarded to `139c8ca`.
- `ops/bin/tp2-install-systemd epc`: installed EPC units.
- `ops/bin/tp2-mqtt-ensure-car-mode`: first run published retained `AM-Cloud`; second run reported retained state already `AM-Cloud` and skipped publishing.
- `tp2-car-control.service`: restarted and active.

## EPC validation

- `bash -n` over ops scripts: ok
- `PYTHONPATH=servicios conda run -n tp2 python -m unittest discover -s tests`: 49 tests ok
- Active listeners: `172.16.0.1:20001/UDP`, `0.0.0.0:8088/TCP`, `0.0.0.0:1883/TCP`
- `/status.json` after safe API posts:
  - `control.mode=manual`, `steering_trim=-0.24`, `applied_steering_trim=-0.24`, `effective_steering=0.01`
  - `autonomy.config.cruise_throttle=0.65`
  - `autonomy.config.left_steering=1.0`, `right_steering=-1.0`
  - `autonomy.turn_compensation.enabled=false`
  - `udp.last_client=172.16.0.4:33453`, live `I` packets, `bad_packets=0`
  - `video.has_video=true`, inference running
- `ops/bin/tp2-status` on EPC: `srsepc`, `mosquitto`, and car control active; S1 established; car UE ok.

## Gap

- Full `ops/bin/tp2-validate` did not complete because SSH to eNodeB `10.10.10.2` timed out. `tp2-status` also reported eNodeB/Jetson SSH unavailable, although the EPC radio/control path was live.
