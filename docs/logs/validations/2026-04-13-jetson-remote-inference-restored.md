# Jetson Remote Inference Restored

- Date: `2026-04-13`
- Machines:
  - `tp2-EPC`
  - `tp2-jetson`
- Scope:
  - restore Jetson inference service after reachability returned
  - validate EPC control remains anchored on EPC
  - validate EPC -> Jetson Roboflow inference path

## Initial State

- Jetson SSH was reachable again at `grupo4@100.115.99.8`
- `tp2-roboflow-inference.service` was `inactive`
- Jetson `127.0.0.1:9001` returned connection refused
- EPC `coche.py` stayed alive on `172.16.0.1:20001`

## Applied Change

- Started only the Jetson inference service:
  - `sudo systemctl start tp2-roboflow-inference.service`
- No firmware, package, LTE, eNodeB, or car-side changes were made
- EPC control process was not restarted

## Runtime Evidence

- Jetson service:
  - `systemctl is-active tp2-roboflow-inference.service` -> `active`
- Jetson container:
  - `tp2-roboflow-inference` -> `Up`
  - image: `roboflow/roboflow-inference-server-jetson-6.2.0:latest`
- Jetson HTTP bind:
  - `ss -ltn | grep 9001` -> `LISTEN 0 2048 0.0.0.0:9001`
- Jetson local HTTP:
  - `curl -fsS --max-time 10 http://127.0.0.1:9001/openapi.json` -> OK
- EPC -> Jetson HTTP:
  - `curl -fsS --max-time 10 http://100.115.99.8:9001/openapi.json` -> OK
- EPC control path:
  - process: `/home/tp2/miniforge3/envs/tp2/bin/python -u coche.py`
  - UDP bind: `172.16.0.1:20001`
  - configured inference endpoint: `ROBOFLOW_LOCAL_API_URL=http://100.115.99.8:9001`
- Known-image inference from EPC to Jetson:
  - script: `/home/tp2/TP2_red4G/servicios/inferencia.py`
  - input: `/home/tp2/TP2_red4G/servicios/test.jpg`
  - output: `/home/tp2/TP2_red4G/servicios/test_pred.jpg`
  - mode: `local`
  - target: `workflow`
  - API URL: `http://100.115.99.8:9001`
  - detections: `1`
  - detected class: `stop sign`

## Result

Jetson remote inference is operational again over Tailscale. The vehicle control process remains on EPC and continues to own the UDP control endpoint for car1.
