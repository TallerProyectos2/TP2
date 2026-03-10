# TP2 Inference Runtime Contract

## Purpose

Define how inference works today (on EPC) and how Jetson will be integrated without breaking the running path.

## Current Production Path (EPC)

Inference is currently available on EPC using scripts from `servicios/`.

- `start_local_inference_server.py`
  - starts a local Roboflow-compatible HTTP interface
  - default bind: `127.0.0.1:9001`
- `inferencia.py`
  - runs inference on a test image
  - supports local or cloud target
  - writes annotated output image
- `inferencia_gui_web.py`
  - batch web GUI for local/cloud inference
  - default bind: `0.0.0.0:7860` (or custom)

## Configuration Contract

Primary environment variables used by scripts:

- `TP2_INFERENCE_MODE` = `local` or `cloud`
- `TP2_INFERENCE_TARGET` = `workflow` or `model`
- `ROBOFLOW_LOCAL_API_URL` (default `http://127.0.0.1:9001`)
- `ROBOFLOW_CLOUD_WORKFLOW_API_URL`
- `ROBOFLOW_CLOUD_MODEL_API_URL`
- `ROBOFLOW_API_KEY`
- `ROBOFLOW_WORKSPACE`
- `ROBOFLOW_WORKFLOW`
- `ROBOFLOW_MODEL_ID`
- `TP2_TEST_IMAGE`
- `TP2_OUTPUT_IMAGE`

## Minimum Validation

1. Start local endpoint on EPC (`start_local_inference_server.py`).
2. Run `inferencia.py` with known image.
3. Confirm:
   - no runtime exception
   - detection JSON is returned
   - annotated image is generated

Evidence reference: `docs/logs/validations/2026-03-05-epc-inferencia-local.md`.

## Jetson Integration Target (Next Phase)

Jetson is pending and should be added as an inference-only node.

Requirements:

- endpoint reachable from EPC
- compatible invocation semantics with current inference client behavior
- switchable from EPC scripts by configuration
- fallback to EPC local inference when Jetson path fails

## Non-Goals In Current Context

- Rebuilding a new inference API stack when existing scripts already cover runtime needs.
- Moving control orchestration away from EPC during Jetson onboarding.
