# Jetson Remote Inference With EPC Control Validation

- Date: `2026-03-26`
- Machine set:
  - `tp2-jetson`
  - `tp2-EPC`
- Scope:
  - Jetson runs Roboflow Inference as HTTP service
  - EPC keeps `coche.py` control ownership for car1 on `172.16.0.1:20001`
  - EPC offloads inference to Jetson over HTTP

## Initial State

- Jetson was reachable again over Tailscale at `100.115.99.8`
- EPC already had a live `coche.py` process bound to `172.16.0.1:20001`
- Jetson endpoint `9001/TCP` was not yet reachable from EPC

## Applied Change

- On Jetson:
  - created `/etc/tp2/jetson-inference.env` with Roboflow runtime secrets
  - installed `/etc/systemd/system/tp2-roboflow-inference.service`
  - started `tp2-roboflow-inference.service`
- On EPC:
  - created `/home/tp2/.config/tp2/coche-jetson.env` with the runtime variables required by `coche.py`
  - relaunched `coche.py` from `/home/tp2/TP2_red4G/servicios`
  - kept bind address and port unchanged: `172.16.0.1:20001`

## Runtime Evidence

- Jetson service state:
  - `ssh grupo4@100.115.99.8 'systemctl is-active tp2-roboflow-inference.service'` -> `active`
- Jetson HTTP bind:
  - `ssh grupo4@100.115.99.8 'ss -ltn | grep 9001'` -> `LISTEN 0 2048 0.0.0.0:9001`
- Jetson HTTP contract:
  - `curl -fsS http://127.0.0.1:9001/openapi.json` on Jetson returned Roboflow OpenAPI JSON
  - `curl -fsS http://100.115.99.8:9001/openapi.json` on EPC returned the same contract
- Known-image inference from EPC to Jetson:
  - command run on EPC:
    - `PYTHONNOUSERSITE=1 conda run -n tp2 env TP2_INFERENCE_MODE=local TP2_INFERENCE_TARGET=workflow ROBOFLOW_LOCAL_API_URL=http://100.115.99.8:9001 ROBOFLOW_WORKSPACE=1-v8mk1 ROBOFLOW_WORKFLOW=custom-workflow-2 ROBOFLOW_API_KEY=*** python inferencia.py`
  - result:
    - `input_image`: `/home/tp2/TP2_red4G/servicios/test.jpg`
    - `output_image`: `/home/tp2/TP2_red4G/servicios/test_pred.jpg`
    - `detections`: `1`
    - detected class: `stop sign`
- EPC live control process after relaunch:
  - `ss -lunp | grep 20001` -> `172.16.0.1:20001` bound by `python`
  - process environment includes:
    - `TP2_INFERENCE_MODE=local`
    - `TP2_INFERENCE_TARGET=workflow`
    - `ROBOFLOW_LOCAL_API_URL=http://100.115.99.8:9001`
    - `ROBOFLOW_WORKSPACE=1-v8mk1`
    - `ROBOFLOW_WORKFLOW=custom-workflow-2`
  - startup log from `/home/tp2/logs/coche-jetson.log`:
    - `Manual control server listening on 172.16.0.1:20001`
    - `Inference: enabled (local/workflow) endpoint=http://100.115.99.8:9001`

## Result

The current validated path is:

1. car attaches over LTE to EPC
2. `coche.py` runs on EPC and owns manual control plus UDP reply path
3. frame inference is offloaded from EPC to Jetson at `100.115.99.8:9001`
4. Jetson does not own orchestration or direct car control

The remaining open item is automatic fallback from Jetson back to EPC local inference when the Jetson endpoint is unavailable.
