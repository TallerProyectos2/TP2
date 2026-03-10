# TP2 System Architecture (Current Operational Context)

## Overview

TP2 runs as a four-machine lab, but the current critical path is script-based and centered on the EPC:

- EPC: LTE core, control scripts, and local inference service.
- eNodeB: radio-only access.
- Car: mobile endpoint sending sensor payloads and receiving control commands.
- Jetson: pending integration for inference offload.

## Current Critical Path

1. Car attaches to LTE and gets UE IP from EPC (`172.16.0.2`).
2. Car sends UDP payloads (image/LIDAR/battery) to EPC control server.
3. EPC script computes steering/throttle.
4. EPC sends UDP control packet back to car.

This path works without introducing a new backend API layer.

## Machine Responsibilities

## PC EPC

- `srsepc` (`MME + HSS + SPGW`)
- UE IP allocation and routing (`172.16.0.0/24`)
- NAT and forwarding
- Optional UE DNS (`dnsmasq`)
- Script runtime from `servicios/`:
  - car control UDP servers
  - local inference endpoint launcher
  - inference CLI and GUI tools

## PC eNodeB

- `srsenb`
- `bladeRF`
- Radio transport only

## Car

- Streams data to EPC over UDP
- Executes control commands received from EPC
- Runs movement logic driven by EPC commands

## Jetson (Pending)

- Planned as inference-only offload node
- Must not host LTE core, DB, MQTT broker, or orchestration

## Network Topology

## EPC <-> eNodeB Backhaul

- `10.10.10.1` (EPC) <-> `10.10.10.2` (eNodeB)
- Carries S1-MME and S1-U

## UE Side

- EPC SGi: `172.16.0.1/24`
- Car UE subnet: `172.16.0.0/24`
- Current fixed car mapping: `901650000052126 -> 172.16.0.2`

## Protocol Contract (Current)

- LTE core transport:
  - `36412/SCTP` (S1-MME)
  - `2152/UDP` (GTP-U)
- Car control transport:
  - UDP script servers on EPC (`20001` for car1 scripts, `20003` for car3 scripts)
  - payload discriminator byte (`I`, `L`, `B`, `D`)
  - control packet type (`C`) with steering/throttle doubles
- Inference transport:
  - local HTTP endpoint (default `127.0.0.1:9001`) for Roboflow-compatible runtime
  - optional cloud endpoint when configured in scripts

## Inference Contract

Inference is currently run on EPC through:

- `start_local_inference_server.py` (local endpoint)
- `inferencia.py` (CLI test and annotated output)
- `inferencia_gui_web.py` (batch GUI)

Jetson integration must preserve compatibility with the current inference client behavior to avoid rewrites of the operational scripts.

## Invariants

- eNodeB remains radio-only.
- EPC remains control and orchestration hub.
- Car does not decide global policy; it executes received commands.
- No firmware upgrades in project operations.
- No parallel rebuild of a new API stack is required to keep the current path operational.
