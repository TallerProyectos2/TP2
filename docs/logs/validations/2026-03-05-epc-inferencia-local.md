# EPC Local Inference Validation

- Date: `2026-03-05`
- Jira: `N/A (Atlassian MCP unavailable in this session)`
- Machine: `tp2-EPC`
- Scope: `inferencia.py` with `test.jpg` executed directly on EPC

## Initial State

- `inferencia.py` existed at `/home/tp2/servicios_tp2/inferencia.py`
- `test.jpg` existed at `/home/tp2/servicios_tp2/test.jpg`
- The `inference-sdk` dependency was not installed on EPC
- The script pointed to `http://localhost:9001` and failed with `Connection refused`

## Applied Change

- Installed `inference-sdk` for user `tp2` with:
  - `python3 -m pip install --user inference-sdk`
- Installed `gradio` for the web GUI with:
  - `python3 -m pip install --user gradio`
- Updated `/home/tp2/servicios_tp2/inferencia.py` to:
  - use an absolute image path based on `__file__`
  - allow configuration through environment variables (`ROBOFLOW_API_URL`, `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, `ROBOFLOW_WORKFLOW`, `TP2_TEST_IMAGE`)
  - default to `https://serverless.roboflow.com`
- Created `/home/tp2/servicios_tp2/inferencia_gui.py` (desktop GUI with multiple selection)
- Created `/home/tp2/servicios_tp2/inferencia_gui_web.py` (web GUI with multiple selection and result gallery)
- Created `/home/tp2/servicios_tp2/start_local_inference_server.py` to start local Roboflow Inference on EPC without Docker (uvicorn + `HttpInterface`)
- Updated `inferencia.py` and `inferencia_gui_web.py` to support switching:
  - mode: `local` or `cloud`
  - target: `workflow` or `model`
  - separate cloud endpoints by target (`serverless` for workflow, `detect` for model)

## Runtime Evidence

- Execution validated on EPC:
  - `cd /home/tp2/servicios_tp2 && python3 inferencia.py`
- Received result:
  - detected class: `stop sign`
  - confidence: `0.9443966746330261`
  - execution completed without exceptions
- Generated visualization:
  - drawn predictions: `1`
  - annotated image: `/home/tp2/servicios_tp2/test_pred.jpg`
  - validated format: `JPEG 1600x1600`
- Web GUI validation:
  - `python3 -m py_compile /home/tp2/servicios_tp2/inferencia_gui_web.py` -> `OK`
  - service startup: `python3 inferencia_gui_web.py --host 127.0.0.1 --port 7861`
  - `ss -ltnp` confirms listening on `127.0.0.1:7861`
- Local/cloud switching validation:
  - cloud+workflow (`TP2_INFERENCE_MODE=cloud`, `TP2_INFERENCE_TARGET=workflow`) runs OK and detects `1` object in `test.jpg`
  - local+workflow (`TP2_INFERENCE_MODE=local`) runs OK and detects `1` object in `test.jpg` after starting the local endpoint
- EPC local endpoint startup:
  - command: `ROBOFLOW_API_KEY=*** python3 /home/tp2/servicios_tp2/start_local_inference_server.py --host 127.0.0.1 --port 9001`
  - state: `ss -ltnp` confirms listening on `127.0.0.1:9001` with PID `49707`
  - HTTP contract: `GET /openapi.json` returns `200` and exposes workflow routes (`/{workspace_name}/workflows/{workflow_id}`)
- Environment note:
  - `tkinter` is not installed on EPC (`ModuleNotFoundError`), so the validated GUI for this state is the web GUI
  - the `inference server start` command from `inference-cli` is not useful on this EPC because it depends on the Docker daemon

## Result

Inference with `test.jpg` is operational by running the script directly on EPC, without depending on Jetson for this test, and now leaves visual evidence with a bounding box and label. EPC also has a web GUI to process one or more images selected from the client filesystem and switch between local and cloud inference.
