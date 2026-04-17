# TP2 Operating Runbook (Script-First)

## Purpose

Define the real startup, validation, and shutdown sequence for the current operational model based on `servicios/`.

## Startup Order

Recommended operator entrypoint:

- From a MacBook with Tailscale access, run `ops/bin/tp2-up jetson` from the repo root.
- The launcher reaches EPC and Jetson directly and reaches the eNodeB through EPC as SSH proxy.

1. Validate EPC <-> eNodeB backhaul connectivity (`10.10.10.1` <-> `10.10.10.2`).
2. Start `srsepc` on EPC.
3. Start `srsenb` on eNodeB.
4. Verify S1 setup is established.
5. Verify car UE attach and IP assignment (`172.16.0.2` expected).
6. Start required script services on EPC:
   - control server script (manual/autonomous/real-time variant)
   - inference endpoint (`start_local_inference_server.py`) if local inference path is needed
7. Optionally start inference GUI web (`inferencia_gui_web.py`) for batch checks.
8. If testing Jetson integration, start Jetson inference service last and enable it via script config.

## Shutdown Order

1. Stop car activity.
2. Stop EPC script services (control server, optional inference GUI/server).
3. Stop `srsenb`.
4. Stop `srsepc`.
5. Collect logs and validation evidence if session changed system state.

## Default Validation Sequence

## LTE Validation

- EPC and eNodeB configs aligned.
- `srsenb` reaches `srsepc`.
- S1 setup completes.
- Car UE attaches and keeps expected IP mapping.

## Script Control Validation

- Selected control script binds its UDP port.
- Script receives car payloads (`I`, `L`, `B`, `D`).
- Script sends control packets (`C`) back to car.
- Car behavior matches command stream.

## EPC Inference Validation

- Local inference endpoint reachable when enabled (default `127.0.0.1:9001`).
- `inferencia.py` runs with a known image and writes annotated output.
- Optional GUI web can process selected images without runtime errors.

## Jetson Validation (when enabled)

- EPC can reach Jetson inference endpoint.
- Script configuration can switch to Jetson target.
- Fallback to EPC local inference is validated.

## Operational Rules

- Prefer read-only inspection first.
- Avoid restarting healthy services unless required by the task.
- Do not change more than one layer at once during troubleshooting.
- If LTE is unstable, stop before script-level or Jetson-level debugging.
- Firmware updates are forbidden on all components.

## Troubleshooting Order

1. EPC and eNodeB process state
2. Backhaul network and S1 state
3. UE attach and IP assignment
4. UDP control script RX/TX path
5. EPC local inference endpoint
6. Jetson inference path (only if enabled)
7. Car-side execution path

## Escalation Conditions

Stop and escalate if:

- credentials are missing,
- remote state is ambiguous,
- a destructive action is required and impact is unclear,
- validation cannot be completed,
- documentation and live system behavior diverge in a way that cannot be explained.
