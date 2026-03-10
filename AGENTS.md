# TP2 Codex Operating Contract

## Mission

This repository operates a four-machine connected-vehicle lab:

- `PC EPC`: LTE core and main runtime host
- `PC eNodeB`: LTE radio access with `bladeRF`
- `Jetson`: pending inference offload node
- `Coche`: sensor/control client over LTE

Future Codex sessions must treat this repository as the control plane for that real lab, not as an isolated code sandbox.

## Mandatory Read Order

Before any non-trivial change, read these files in order:

1. `PLAN.md`
2. `ARCHITECTURE.md`
3. `RUNBOOK.md`
4. `MACHINES.md`
5. The relevant service or machine runbook under `docs/`

## Source Of Truth

- `PLAN.md`: canonical implementation plan
- `ARCHITECTURE.md`: system boundaries and data flow
- `RUNBOOK.md`: startup, shutdown, and operation sequence
- `MACHINES.md`: ownership, addressing, and access path

Do not silently diverge from these documents. If the real system changes, update docs in the same task.

## Architecture Contract (Current)

- eNodeB remains radio-only.
- EPC is the runtime hub for LTE and script orchestration.
- Car exchanges control data with EPC scripts over UDP.
- Inference is currently available on EPC via scripts in `servicios/`.
- Jetson is integrated only as inference offload, never as orchestration host.
- No firmware updates on any component.

## Default Delivery Loop

For any non-trivial task:

1. Identify the highest-priority unblocked item in `PLAN.md`.
2. Inspect the current state first:
   - local repo state
   - relevant remote machine state
3. Implement the change end-to-end for affected machine/service.
4. Validate with runtime checks.
5. Update docs if operating model changed.
6. Add evidence in `docs/logs/validations/` for material changes.

Do not mark work complete based on intent alone.

## Remote Machine Rules

- Prefer read-only checks first.
- Use the established SSH path to EPC first, then hop to eNodeB if needed.
- Do not store passwords, tokens, or secrets in repository files.
- Do not restart already-working services unless required by the task.
- Never update firmware on any component under any circumstance.
- For risky remote actions, stop and ask if blast radius is unclear.

## Validation Minimums

- LTE work:
  - config check
  - process check
  - port check
  - reachability check
- EPC script runtime work:
  - process/port check of selected script
  - control RX/TX path check
  - required dependency check
- Inference work:
  - endpoint health/reachability
  - known-image inference check
- Car runtime work:
  - command reception
  - movement/control execution path
  - safe fallback behavior

End-to-end changes should include the active real path:

- UE attach
- sensor payload reception
- inference/control computation
- command response to car
- observed execution behavior

## Non-Negotiables

- Never commit secrets.
- Never update firmware on `bladeRF`, modem, Jetson, car, or any other component.
- Never write SSH passwords into docs, scripts, or comments.
- Never move eNodeB into hosting application/orchestration services.
- Never claim a phase complete without runtime evidence.

## Stop Conditions

Stop and escalate when:

- credentials are missing,
- remote machine state is ambiguous,
- a destructive action is required,
- validation cannot be completed,
- documented architecture and live technical state disagree and the cause is unclear.
