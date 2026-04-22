# 2026-04-14 - Jetson local Roboflow model `tp2-g4-2026/2`

## Scope

Move the live inference path from Roboflow workflow invocation to direct model invocation while preserving the existing control architecture:

- EPC remains the control and orchestration host.
- Jetson remains inference-only.
- Car control stays on EPC UDP scripts.
- Secrets stay in machine-local env files, not in the repository.

## Changes Applied

- Updated `servicios/coche.py` defaults:
  - `TP2_INFERENCE_MODE=local`
  - `TP2_INFERENCE_TARGET=model`
  - `ROBOFLOW_LOCAL_API_URL=http://100.115.99.8:9001`
  - `ROBOFLOW_MODEL_ID=tp2-g4-2026/2`
- Updated EPC machine-local runtime env:
  - `/home/tp2/.config/tp2/coche-jetson.env`
  - `/home/tp2/.config/tp2/inference.env -> coche-jetson.env`
- Updated EPC Conda activation hook:
  - `/home/tp2/miniforge3/envs/tp2/etc/conda/activate.d/tp2-runtime.sh`
- Started and enabled Jetson systemd service:
  - `tp2-roboflow-inference.service`

## Validation

- EPC SSH:
  - command: `ssh tp2@100.97.19.112 hostname`
  - result: `tp2-EPC`
- Jetson SSH:
  - command: `ssh grupo4@100.115.99.8 hostname`
  - result: `tp2-jetson`
- Jetson service:
  - `systemctl is-active tp2-roboflow-inference.service` -> `active`
  - `systemctl is-enabled tp2-roboflow-inference.service` -> `enabled`
  - `ss -ltn | grep 9001` -> `LISTEN 0.0.0.0:9001`
- Jetson endpoint:
  - command: `curl -fsS --max-time 5 http://100.115.99.8:9001/info`
  - result: `{"name":"Roboflow Inference Server","version":"1.1.2","uuid":"JBDdvZ-GPU-0"}`
- EPC Conda runtime after `conda activate tp2`:
  - `TP2_ACTIVE_ENV_FILE=/home/tp2/.config/tp2/inference.env`
  - `TP2_INFERENCE_MODE=local`
  - `TP2_INFERENCE_TARGET=model`
  - `ROBOFLOW_LOCAL_API_URL=http://100.115.99.8:9001`
  - `ROBOFLOW_MODEL_ID=tp2-g4-2026/2`
  - `ROBOFLOW_API_KEY=<set>`
- EPC CLI inference:
  - command: `conda activate tp2 && cd /home/tp2/TP2_red4G/servicios && TP2_OUTPUT_IMAGE=/tmp/tp2_model_pred.jpg python inferencia.py`
  - result:
    - `mode=local`
    - `target=model`
    - `api_url=http://100.115.99.8:9001`
    - `detections=1`
    - detected class: `STOP`
    - confidence: `0.9261639714241028`
- EPC live worker import test:
  - `WORKER_STATUS=ok`
  - `WORKER_TARGET=model`
  - `WORKER_MODEL_ID=tp2-g4-2026/2`
  - `WORKER_API_URL=http://100.115.99.8:9001`
  - `WORKER_DETECTIONS=1`
  - `WORKER_ERROR_PRESENT=False`
- `coche.py` startup config check:
  - command used `TP2_BIND_IP=127.0.0.1` because the live `172.16.0.1` SGi interface was not present at validation time.
  - output included: `Inference: enabled (local/model) endpoint=http://100.115.99.8:9001`
- EPC Python compile:
  - command: `python -m py_compile coche.py roboflow_runtime.py inferencia.py`
  - result: OK

## Notes

- Direct bind to `172.16.0.1:20001` failed during this validation with `OSError: [Errno 99] Cannot assign requested address`, which is consistent with the EPC LTE SGi interface not being up at the time of this check. For real car operation, start the LTE/EPC path first so `172.16.0.1` exists, then run `coche.py`.
- No Roboflow token or SSH password was written to repository files.
