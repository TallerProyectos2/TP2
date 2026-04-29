# 2026-04-29 - Lane assist bias correction hardening

## Scope

- Kept EPC as the only car control runtime.
- Hardened `servicios/lane_detector.py` for the live taped track where the right lane line can exit the camera frame.
- Preserved the steering convention: smaller steering values turn right, so lane corrections and `TP2_STEERING_TRIM=-0.08` compensate the physical left drift to the right.
- Prevented stale lane memory from refreshing itself indefinitely.
- Lane assist still does not apply during open turn actions.
- No firmware changes.

## Validation

- Local real-frame replay using EPC captures from `/srv/tp2/frames/autonomous/20260429-092908/images/`:
  - Before: `no-plausible-lane-width`.
  - After: `pair`, `confidence=0.84`, `correction=-0.05` on sampled frames.
- Local MacBook:
  - `PYTHONPATH=servicios python -m compileall -q servicios tests`
  - `PYTHONPATH=servicios python -m unittest discover -s tests`
  - Result: 39 tests OK.

## EPC note

At initial inspection `tp2-car-control.service` was inactive, while `tp2-srsepc.service` and `mosquitto.service` were active. After deploy:

- EPC pulled `main` to `3a1dd5e`.
- EPC tests passed: 39 tests OK.
- `sudo -n systemctl start tp2-car-control.service` started the runtime.
- `172.16.0.1:20001/UDP` and `0.0.0.0:8088/TCP` were listening.
- Synthetic UDP frame check returned `C`, steering `0.17`, throttle `0.0` in manual neutral.
- Live car frames then arrived from `172.16.0.2`; `/status.json` showed `lane.status=tracking`, `lane.usable=true`, `lane.guidance.source=pair`, `lane.guidance.correction=-0.117`.
- `tp2-srsepc.service`, `mosquitto.service`, and `tp2-car-control.service` were active.
