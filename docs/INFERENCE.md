# TP2 Inference Runtime Contract

## Purpose

Define how inference works today with EPC-owned control and switchable Roboflow inference backends.

## Current Production Path

Inference is currently available on EPC using scripts from `servicios/`.

- `start_local_inference_server.py`
  - starts a local Roboflow-compatible HTTP interface
  - default bind: `127.0.0.1:9001`
- `inferencia.py`
  - runs inference on a test image
  - supports local or cloud target
  - writes annotated output image
- `coche.py`
  - live car-control runtime on EPC
  - browser control updates EPC state while UDP control remains on EPC
  - can call a Roboflow-compatible endpoint for frame inference while the UDP control loop stays anchored on EPC
  - defaults live inference to Jetson at `http://100.115.99.8:9001`
  - default Jetson target is direct model inference with `ROBOFLOW_MODEL_ID=tp2-g4-2026/2`
  - exposes annotated live video, browser control, and inference status on `0.0.0.0:8088` for Tailscale operators

Configured backend options:

- EPC local endpoint at `127.0.0.1:9001`
- Jetson remote endpoint at `100.115.99.8:9001` when reachable

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

Machine-local persistent env files:

- EPC preferred path: `/home/tp2/.config/tp2/inference.env`
- EPC compatibility path: `/home/tp2/.config/tp2/coche-jetson.env`
- These files may contain secrets and must not be copied into the repository.
- `roboflow_runtime.py` loads these files automatically before reading defaults.
- `conda activate tp2` also loads the same runtime through `/home/tp2/miniforge3/envs/tp2/etc/conda/activate.d/tp2-runtime.sh`.

## Minimum Validation

1. Start local endpoint on EPC (`start_local_inference_server.py`).
2. Run `inferencia.py` with known image.
3. Confirm:
   - no runtime exception
   - detection JSON is returned
   - annotated image is generated

Evidence references:

- `docs/logs/validations/2026-03-05-epc-inferencia-local.md`
- `docs/logs/validations/2026-03-26-jetson-remote-inference-epc-control.md`
- `docs/logs/validations/2026-04-13-jetson-remote-inference-restored.md`
- `docs/logs/validations/2026-04-13-coche-defaults-to-jetson-inference.md`
- `docs/logs/validations/2026-04-13-persistent-roboflow-token-machine-env.md`
- `docs/logs/validations/2026-04-13-conda-activate-tp2-runtime-env.md`
- `docs/logs/validations/2026-04-14-jetson-local-roboflow-model-tp2-g4-2026-2.md`

## Jetson Offload Path
Jetson is integrated as an inference-only node.

Reference runbook:

- `docs/JETSON.md`

Requirements:

- endpoint reachable from EPC
- compatible invocation semantics with current inference client behavior
- switchable from EPC scripts by configuration
- fallback to EPC local inference when Jetson path fails
- car continues talking only to EPC, never directly to Jetson

Last validated Jetson configuration:

- Jetson service: `tp2-roboflow-inference.service`
- Jetson endpoint from EPC: `http://100.115.99.8:9001`
- EPC target: `TP2_INFERENCE_TARGET=model`
- Roboflow model: `ROBOFLOW_MODEL_ID=tp2-g4-2026/2`
- EPC live control process: `coche.py` on `172.16.0.1:20001`
- EPC live operator view: `http://100.97.19.112:8088/`
- Runtime secrets stay outside the repository (for example host-local env files)

Live availability is not assumed. Recheck Jetson reachability before starting a session with remote inference enabled.

## Non-Goals In Current Context

- Rebuilding a new inference API stack when existing scripts already cover runtime needs.
- Moving control orchestration away from EPC.
