# TP2 Reliability Model (Current)

## Reliability Goals

- LTE starts repeatably (`srsepc` + `srsenb`).
- Car UE attach remains stable with expected UE IP mapping.
- EPC control scripts maintain UDP RX/TX loop with the car.
- Inference on EPC remains callable and deterministic.
- Jetson integration does not break EPC fallback path.

## Primary Failure Domains

- EPC/eNodeB config drift
- `bladeRF` startup instability
- UE attach/auth provisioning mismatch
- UDP control script not bound or blocked
- inference endpoint down (`127.0.0.1:9001` when required)
- Jetson endpoint unreachable when enabled

## Fallback Rules

- If Jetson inference fails:
  - fallback to EPC local inference target.

- If local inference endpoint fails:
  - hold or reduce control aggressiveness and avoid stale command loops.

- If autonomous mode loses fresh frames or inference:
  - return steering/throttle to neutral and surface `autonomous-safe` in web status.

- If autonomous detections are ambiguous:
  - hold neutral or crawl rather than choosing a conflicting maneuver.
  - expose the selected state/reason in `/status.json` for operator diagnosis.

- If dataset recording fails:
  - keep control runtime alive and surface recorder `last_error`; recording is diagnostic, not part of the control safety path.

- If UDP control loop breaks:
  - car must fall back to safe stop behavior.

## Operational Logging

Capture evidence for:

- LTE process and port state
- UE attach and IP assignment
- control script startup and packet flow
- inference request outcomes
- fallback trigger events

## Troubleshooting Strategy

Fix one layer at a time:

1. LTE transport
2. UE routing and attach
3. EPC UDP control loop
4. EPC inference endpoint
5. Jetson inference path
6. Car execution path
