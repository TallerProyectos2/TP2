# Context Rebaseline To Script-First Model

- Date: `2026-03-10`
- Scope: repository planning/context update

## Trigger

Operational context changed:

- inference model/service is available on EPC,
- existing scripts under `servicios/` are the working runtime baseline,
- EPC + eNodeB + car are considered configured,
- Jetson integration is the main pending implementation block.

## Evidence Reviewed

- Existing Codex validation log:
  - `docs/logs/validations/2026-03-05-epc-inferencia-local.md`
- Latest script bundle commit:
  - `61a9b85` (`scripts IA`) with `servicios/*.py`
- Current LTE/UE context:
  - `docs/logs/validations/2026-03-10-car-ue-ip-assignment.md`

## Repo Changes Applied

- Reorganized global plan:
  - `PLAN.md`
- Updated core source-of-truth context:
  - `ARCHITECTURE.md`
  - `RUNBOOK.md`
  - `MACHINES.md`
- Updated machine/network/runtime docs:
  - `docs/EPC.md`
  - `docs/NETWORK.md`
  - `docs/INFERENCE.md`
  - `docs/CAR-AGENT.md`
  - `docs/DESIGN.md`
  - `docs/RELIABILITY.md`

## Result

Repository context now reflects the real operational model:

- script-first runtime on EPC,
- no mandatory new backend API in current critical path,
- Jetson tracked as next integration stage without breaking the validated EPC+eNodeB+car baseline.
