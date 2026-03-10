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
