# Jetson Unreachable During Remote Inference Follow-Up

- Date: `2026-04-13`
- Scope:
  - Validate live EPC -> Jetson inference route after previous `2026-03-26` integration
  - Avoid changing EPC control process unless Jetson route is available

## Checks

- EPC control process:
  - command: `pgrep -af "python.*coche.py"`
  - result: `/home/tp2/miniforge3/envs/tp2/bin/python -u coche.py`
- EPC UDP bind:
  - command: `ss -lunp | grep 20001`
  - result: `172.16.0.1:20001` bound by `python`
- EPC -> Jetson Tailscale HTTP:
  - command: `curl -fsS --max-time 5 http://100.115.99.8:9001/openapi.json`
  - result: timeout
- Direct SSH to Jetson Tailscale:
  - command: `ssh grupo4@100.115.99.8 'hostname'`
  - result: operation timed out
- EPC -> Jetson management LAN:
  - command: `ping -c 1 -W 1 192.168.72.127`
  - result: failed
  - command: `curl -fsS --max-time 5 http://192.168.72.127:9001/openapi.json`
  - result: timeout
- EPC Tailscale view:
  - `tp2-jetson` shown as offline, last seen `17d ago`

## Result

The EPC control path remains up, but Jetson remote inference is not currently reachable. Keep control anchored on EPC. Do not treat Jetson offload as available until the Jetson is powered/networked and both SSH and `9001/TCP` pass validation again.

## Resolution

- Date: `2026-04-13`
- Jetson became reachable again over Tailscale at `100.115.99.8`
- `tp2-roboflow-inference.service` was found `inactive` and started manually
- After container startup, Jetson listened on `0.0.0.0:9001`
- EPC reached `http://100.115.99.8:9001/openapi.json`
- EPC `inferencia.py` completed successfully against Jetson with `1` detection on `test.jpg`
- Recovery evidence: `docs/logs/validations/2026-04-13-jetson-remote-inference-restored.md`
