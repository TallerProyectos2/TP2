# Car Runtime Contract (Current)

## Purpose

Document the current car-side interaction model used by existing EPC scripts.

## Control Model

The active operational model is UDP stream + UDP control:

- Car sends payloads to EPC control server.
- EPC script processes payloads and returns control command.

No new backend API is required to run this path.

## Runtime Endpoint In Use

Normal sessions use one EPC runtime from `servicios/`:

- `coche.py`

## UDP Packet Contract (As Implemented)

- Incoming payload discriminator (first byte):
  - `I`: camera image payload
  - `B`: battery level
  - `D`: reserved/other data path
- Payload body is deserialized with `pickle.loads(...)` in current scripts.
- Outgoing control packet type:
  - `C` + steering (`double`) + throttle (`double`)

## Runtime Modes

- Web manual mode:
  - browser-driven steering/throttle through `coche.py` on EPC
  - camera display for operator feedback
  - optional Roboflow inference overlay on live camera frames

## LTE Binding Context

- Car attaches as UE in EPC network.
- Current static mapping:
  - IMSI `901650000052126` -> `172.16.0.2`

## Minimum Validation

1. Start LTE (`srsepc` + `srsenb`) and verify UE attach.
2. Start `tp2-car-control.service` on EPC.
3. Confirm script receives UDP payloads from car.
4. Confirm control packets are returned and car responds.

## EPC Operator Notes

- Current LTE-side runtime bind is `172.16.0.1:20001`.
- `coche.py` is the only normal car runtime in `servicios/`.
- `coche.py` exposes the live camera/inference/operator status and remote manual control web view on `8088/TCP`; use `http://100.97.19.112:8088/` from Tailscale during normal EPC-run sessions.
- Browser control has a watchdog: if the web UI stops sending commands, EPC returns to neutral instead of holding the last throttle.
- `coche.py` defaults its live inference endpoint to Jetson at `http://100.115.99.8:9001` using direct model inference (`TP2_INFERENCE_TARGET=model`, `ROBOFLOW_MODEL_ID=tp2-g4-2026/2`); override these variables only when intentionally using another backend.
- `coche.py` loads `/home/tp2/.config/tp2/inference.env` or `/home/tp2/.config/tp2/coche-jetson.env` automatically, so operators do not need to `source` the token manually.
- On the EPC, `conda activate tp2` also loads the same runtime variables and token through the Conda activation hook.
- The validated Jetson offload path keeps EPC as control owner and points inference to `ROBOFLOW_LOCAL_API_URL=http://100.115.99.8:9001` when Jetson is reachable.
- Secrets required for Roboflow access must stay in host-local env files or shell state, never in the repository.
