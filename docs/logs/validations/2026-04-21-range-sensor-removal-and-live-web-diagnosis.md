# 2026-04-21 - Range-sensor removal and live web diagnosis

## Scope

- Removed range-sensor-specific code paths and documentation references because the current car only uses camera, battery, and runtime payloads.
- Kept EPC-owned UDP control and Jetson-owned inference unchanged.
- Inspected the live lab state from the MacBook without restarting services.

## Local Checks

- Range-sensor string search across code, docs, and root markdown.
  - result: no matches.
- Removed packet-type `L` handling checks across service scripts.
  - result: no matches.
- `python3 -m py_compile` on modified Python service files.
  - result: success.

## Remote Read-Only Checks

- `ops/bin/tp2-status`
  - EPC `srsepc`: active.
  - eNodeB `srsenb`: active.
  - S1 association: established.
  - car UE: confirmed at `172.16.0.2`.
  - EPC UDP control listener: `172.16.0.1:20001`.
  - EPC live web listener: `0.0.0.0:8088`.
  - Jetson inference service: active.
  - Jetson OpenAPI check: ok.

- `curl http://100.97.19.112:8088/status.json`
  - result: web server responded.
  - observed `has_video=false`.
  - observed `video_frames=0`.
  - observed inference status `waiting`.

- EPC to Jetson reachability:
  - command from EPC: `curl http://100.115.99.8:9001/openapi.json`
  - result: ok.

## Diagnosis

The live web server is reachable and the Jetson inference endpoint is reachable from EPC. The missing video and inference are therefore not explained by Jetson reachability. The active `coche.py` process has not received camera frames since service start, so the web view has no image to publish and the inference worker remains waiting.

Likely next checks:

- confirm the car is sending image UDP payloads to `172.16.0.1:20001`;
- confirm the car mode that is published over MQTT also starts the camera streaming path;
- if packet capture is needed on EPC, run it with an operator-provided sudo path because passwordless packet capture is not configured.
