# TP2 Connected Vehicle LTE & Edge AI Lab

TP2 is a connected-vehicle lab that combines LTE networking, real-time control, perception tooling, and edge-inference preparation across real hardware.

The repository documents and implements a script-first operating model built around four machine roles:

- EPC: LTE core, control runtime, and current inference host
- eNodeB: radio access only
- Car: LTE-connected endpoint sending payloads and receiving commands
- Jetson Orin Nano: planned inference-only offload node

This is not a production autonomous-driving stack. It is a validated lab prototype designed to test how radio, control, inference, and operational tooling behave together under real networking constraints.

## What the project does

- connects the vehicle to the lab over LTE
- receives image, battery, and runtime payloads on the EPC
- computes steering and throttle decisions from EPC-side control scripts
- sends commands back to the car over UDP
- exposes live camera, inference overlays, and remote manual control from the EPC web view
- supports local and cloud inference workflows for perception experiments
- prepares a clean path for Jetson-based inference offload without breaking the validated EPC-centric runtime
- includes operator runbooks, machine inventory, validation logs, and bring-up automation

## Current operational architecture

```text
Car (UE)
  -> LTE attach
  -> eNodeB
  -> EPC

Car -> UDP payloads (image / battery / runtime) -> EPC
EPC -> control decisions (steering / throttle) -> Car

Jetson Orin Nano
  -> planned inference-only node
  -> reachable from EPC
  -> optional offload target, not the orchestration hub
```

Current validated path:

1. The car attaches to LTE and receives a UE address from the EPC.
2. The car sends UDP payloads to the EPC control server.
3. EPC-side scripts compute steering and throttle.
4. The EPC sends UDP control packets back to the car.

This path is intentionally simple and operationally clear. The system is built around what already works on real infrastructure instead of introducing an unnecessary API layer.

## Repository structure

- [`servicios/`](./servicios): runtime scripts for control, local/cloud inference, operator GUI tools, and supporting logic
- [`ops/`](./ops): lab orchestration scripts, environment template, systemd units, sudoers files, and shared shell helpers
- [`docs/`](./docs): focused documentation for network, EPC, inference, Jetson, security, reliability, and validation logs
- [`ARCHITECTURE.md`](./ARCHITECTURE.md): current machine responsibilities and runtime contract
- [`RUNBOOK.md`](./RUNBOOK.md): startup, validation, troubleshooting, and shutdown order
- [`MACHINES.md`](./MACHINES.md): current machine inventory and role ownership
- [`scripts/`](./scripts): supporting setup scripts

## Operating model

This repository assumes access to the real lab machines and network. It is not meant to be cloned and run end-to-end on an arbitrary laptop with no hardware.

The recommended entrypoint is:

1. Read [`ARCHITECTURE.md`](./ARCHITECTURE.md), [`RUNBOOK.md`](./RUNBOOK.md), and [`MACHINES.md`](./MACHINES.md).
2. Copy environment defaults from [`ops/tp2-lab.env.example`](./ops/tp2-lab.env.example) into a local override file.
3. Install the required sudoers and systemd units on the target machines if needed.
4. Bring the lab up in the documented order.
5. Validate LTE, control, and optional inference paths before changing anything else.

Useful operator commands in [`ops/bin/`](./ops/bin):

- `tp2-up`: bring up the lab path in the expected order
- `tp2-status`: inspect current system state
- `tp2-validate`: run core validation checks
- `tp2-down`: stop the lab cleanly
- `tp2-install-sudoers`: install scoped sudoers files
- `tp2-install-systemd`: install systemd units on target machines

## Inference and Jetson path

Inference runs on the EPC today through the scripts in [`servicios/`](./servicios), including:

- local Roboflow-compatible inference endpoint
- CLI inference validation
- GUI / web tooling for batch inspection

Jetson integration is the next phase, but with an important constraint: Jetson should remain an inference-only offload node. EPC stays responsible for LTE core functions, orchestration, and the active control path.

Relevant docs:

- [`docs/INFERENCE.md`](./docs/INFERENCE.md)
- [`docs/JETSON.md`](./docs/JETSON.md)

## Documentation map

- [`docs/NETWORK.md`](./docs/NETWORK.md): network contract, ports, and reachability
- [`docs/EPC.md`](./docs/EPC.md): EPC-specific notes
- [`docs/CAR-AGENT.md`](./docs/CAR-AGENT.md): car-side behavior
- [`docs/RELIABILITY.md`](./docs/RELIABILITY.md): operational reliability principles
- [`docs/SECURITY.md`](./docs/SECURITY.md): security notes
- [`docs/logs/index.md`](./docs/logs/index.md): validation evidence and operating history

## Current maturity

TP2 should be read as a serious lab environment and engineering testbed:

- real hardware
- real LTE transport
- real control traffic
- real machine coordination
- real operational validation logs

It is not yet a production platform, and Jetson offload is still a controlled integration phase rather than the default runtime.

## Public repository note

This repository is public. Do not commit:

- passwords
- API tokens
- private SSH material
- machine-specific secret overrides

Store real secrets in local environment files or target-machine configuration outside the repository. Use [`ops/tp2-lab.env.example`](./ops/tp2-lab.env.example) as a template only.
