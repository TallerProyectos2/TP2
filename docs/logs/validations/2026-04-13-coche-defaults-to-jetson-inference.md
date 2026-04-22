# Coche Script Defaults To Jetson Inference

- Date: `2026-04-13`
- Machine: `tp2-EPC`
- Scope:
  - make `servicios/coche.py` default live inference to Jetson
  - keep EPC as UDP control owner for car1

## Applied Change

- Updated `servicios/coche.py` so the car live runtime sets these defaults before importing the shared Roboflow runtime:
  - `TP2_INFERENCE_MODE=local`
  - `TP2_INFERENCE_TARGET=workflow`
  - `ROBOFLOW_LOCAL_API_URL=http://100.115.99.8:9001`
  - `ROBOFLOW_WORKSPACE=1-v8mk1`
  - `ROBOFLOW_WORKFLOW=custom-workflow-2`
- Secrets remain external to the repository.
- Deployed the updated script to `/home/tp2/TP2_red4G/servicios/coche.py` on the EPC.

## Runtime Evidence

- Remote syntax check:
  - `conda run -n tp2 python -m py_compile coche.py` -> OK
- Existing car1 control runtime restarted on EPC:
  - process: `/home/tp2/miniforge3/envs/tp2/bin/python -u coche.py`
  - PID during validation: `3866`
  - UDP bind: `172.16.0.1:20001`
- Runtime environment:
  - `TP2_INFERENCE_MODE=local`
  - `TP2_INFERENCE_TARGET=workflow`
  - `ROBOFLOW_LOCAL_API_URL=http://100.115.99.8:9001`
  - `ROBOFLOW_WORKSPACE=1-v8mk1`
  - `ROBOFLOW_WORKFLOW=custom-workflow-2`
- Startup log:
  - `Inference: enabled (local/workflow) endpoint=http://100.115.99.8:9001`
- Default-without-env check:
  - command cleared the inference endpoint env vars and launched `coche.py` on test UDP port `29999`
  - startup log still printed `endpoint=http://100.115.99.8:9001`

## Result

Starting `coche.py` now uses Jetson for live inference by default. If it prints `127.0.0.1:9001`, the operator shell or service has explicitly set `ROBOFLOW_LOCAL_API_URL` and that environment variable should be unset or changed.
