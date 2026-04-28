# TP2 Design Notes (Current)

## Functional Design

The current system is intentionally script-first:

- LTE transport is isolated in EPC + eNodeB.
- Control loop runs on EPC scripts.
- Car acts as sensor/control endpoint.
- Inference runs on EPC today, with Jetson planned as optional offload.

## Why EPC Is The Runtime Hub

EPC already owns:

- LTE core state
- UE routing
- central operator access path

Keeping script orchestration there reduces moving parts and avoids introducing a second orchestration layer prematurely.

## Why No New API Layer Right Now

Existing scripts in `servicios/` already provide:

- car control loop
- autonomous/manual operation modes
- local/cloud inference tooling

Building a new API stack now would duplicate behavior and slow integration without reducing current risk.

## Why Jetson Is Deferred To Integration Phase

Jetson is still useful, but should be added only as:

- inference offload endpoint
- configuration-selectable target
- fallback-safe extension

This avoids destabilizing the validated EPC+eNodeB+car path.

## Data/Transport Design (Current)

- LTE:
  - attach and user plane through EPC/eNodeB
- Car control:
  - UDP payload stream to EPC scripts
  - UDP control response from EPC to car
  - mode selection stays in EPC web runtime; manual browser commands and autonomous decisions both produce the same `C + steering + throttle` packet
- Inference:
  - local endpoint on EPC (or cloud target when configured)
  - detections feed the EPC autonomous policy; Jetson remains an inference endpoint only
- Dataset capture:
  - EPC records candidate frames and estimates for later curation
  - recorded estimates are not treated as ground-truth labels until manually reviewed

## Design Invariants

- eNodeB stays radio-only.
- Control decisions stay centralized on EPC runtime.
- Jetson is inference-only when integrated.
- No firmware upgrades in project operations.
