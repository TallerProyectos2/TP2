# Persistent Roboflow Token For EPC Runtime

- Date: `2026-04-13`
- Machine: `tp2-EPC`
- Scope:
  - fix Jetson inference `401 Unauthorized` caused by missing client API key
  - keep the Roboflow token persistent on the machine, not in the repository
  - apply the same persistent config to `coche.py`, `inferencia.py`, and `inferencia_gui_web.py`

## Applied Change

- Updated `servicios/roboflow_runtime.py` to load machine-local env files before resolving Roboflow defaults:
  - `/home/tp2/.config/tp2/inference.env`
  - `/home/tp2/.config/tp2/coche-jetson.env`
  - `/etc/tp2/inference.env`
  - `/etc/tp2/coche-jetson.env`
- Updated `servicios/coche.py` to load its machine-local env before importing `pynput`, so `DISPLAY` and input handling are initialized correctly.
- Deployed updated scripts to `/home/tp2/TP2_red4G/servicios/` on EPC.
- Created `/home/tp2/.config/tp2/inference.env` as a symlink to the existing machine-local `/home/tp2/.config/tp2/coche-jetson.env`.
- Preserved file permissions:
  - `/home/tp2/.config/tp2/coche-jetson.env` -> `600 tp2:tp2`

## Runtime Evidence

- Persistent token presence was checked without printing the secret:
  - `PERSISTENT_API_KEY_LEN:20`
- Common runtime load check with Roboflow env vars cleared:
  - `RUNTIME_API_URL:http://100.115.99.8:9001`
  - `RUNTIME_API_KEY_LEN:20`
  - `RUNTIME_TARGET:workflow`
  - `RUNTIME_WORKFLOW:custom-workflow-2`
- `inferencia.py` with Roboflow env vars cleared:
  - API URL: `http://100.115.99.8:9001`
  - detections: `1`
  - detected class: `stop sign`
- `coche.py` worker validation with Roboflow env vars cleared:
  - `WORKER_STATUS:ok`
  - `WORKER_DETECTIONS:1`
  - `WORKER_ERROR_PRESENT:False`
- Live EPC control process after restart:
  - UDP bind: `172.16.0.1:20001`
  - startup log: `Inference: enabled (local/workflow) endpoint=http://100.115.99.8:9001`

## Result

The Roboflow token is now persistent on the EPC and loaded automatically by the runtime. Starting `coche.py` directly no longer requires manually exporting or sourcing the API key, and the Jetson inference path no longer fails with `401 Unauthorized` when using the machine-local configuration.
