# TP2 System Architecture (Current Operational Context)

## Overview

TP2 runs as a four-machine lab, but the current critical path is script-based and centered on the EPC:

- EPC: LTE core, control scripts, and local inference service.
- eNodeB: radio-only access.
- Car: mobile endpoint sending sensor payloads and receiving control commands.
- Jetson: inference-only offload node when reachable.

## Current Critical Path

1. Car attaches to LTE and gets UE IP from EPC. The previous fixed target was `172.16.0.2`; the live EPC HSS was observed on `2026-04-27` with dynamic allocation and the active session at `172.16.0.4`.
2. Car sends UDP payloads (image/battery/runtime) to EPC control server.
3. EPC script computes steering/throttle in either manual web mode or autonomous mode. In autonomous mode, traffic-sign decisions remain Roboflow-driven and `lane_detector.py` adds a bounded OpenCV lane correction from the blue/green tape lines when the car is moving forward.
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

## Jetson

- Integrated as inference-only offload node
- Exposes a Roboflow-compatible HTTP endpoint on `9001/TCP`
- Last validated EPC-reachable endpoint: `http://100.115.99.8:9001`
- Current EPC-selected Roboflow target is direct model inference: `tp2-g4-2026/2`
- Must not host LTE core, DB, MQTT broker, or orchestration

## Network Topology

## EPC <-> eNodeB Backhaul

- `10.10.10.1` (EPC) <-> `10.10.10.2` (eNodeB)
- Carries S1-MME and S1-U

## UE Side

- EPC SGi: `172.16.0.1/24`
- Car UE subnet: `172.16.0.0/24`
- Previous fixed car mapping target: `901650000052126 -> 172.16.0.2`
- Live note (`2026-04-27`): EPC HSS currently has `IP_alloc=dynamic` for the car IMSI, so the active UE IP can change.

## Protocol Contract (Current)

- LTE core transport:
  - `36412/SCTP` (S1-MME)
  - `2152/UDP` (GTP-U)
- Car control transport:
  - UDP runtime on EPC (`172.16.0.1:20001`)
  - payload discriminator byte (`I`, `B`, `D`)
  - control packet type (`C`) with steering/throttle doubles
- Operator visibility:
  - `coche.py` exposes annotated live video, remote manual control, autonomous mode, and inference/control status from EPC on `8088/TCP`
  - browser control updates EPC state only; EPC remains the only host that sends UDP commands to the car
  - steering compensation, autonomous cruise throttle, and optional periodic right-turn pulse are mutable from the EPC web UI; steering trim is applied immediately before UDP command serialization and bypassed only for autonomous open turns
  - autonomous driving is an EPC-local decision layer over Roboflow detections; it performs temporal tracking, sign selection, stateful maneuvers, and command smoothing without moving orchestration to Jetson or the car
  - lane assistance is EPC-local OpenCV processing over the same camera frames; it detects the blue/green tape corridor on the black carpet and reports status/correction through `/status.json`
  - session recording is also EPC-local and stores candidate frames, annotated MP4 video, predictions, critical flags, reviewed-label sidecars, and autonomous estimates for dataset improvement under the configured recording directory
  - offline session replay/relabeling stays in `servicios/session_replayer.py`; it does not become a runtime service in the control path
- Inference transport:
  - local HTTP endpoint (default `127.0.0.1:9001`) for Roboflow-compatible runtime
  - optional Jetson HTTP endpoint (`100.115.99.8:9001` when reachable)
  - optional cloud endpoint when configured in scripts

## Inference Contract

Inference is consumed by EPC through:

- `coche.py` (live frame sender and annotation owner)
- `autonomous_driver.py` (deterministic traffic-sign controller used by `coche.py`: tracker, distance proxy, FSM, safety fallback)
- `lane_detector.py` (classical OpenCV lane-corridor detector used only as steering stabilization in `coche.py`)
- `inferencia.py` (CLI test and annotated output)
- `start_local_inference_server.py` (optional EPC local fallback endpoint)

Jetson offload preserves compatibility with the current inference client behavior to avoid rewrites of the operational scripts. Its live reachability must be validated before enabling it for a session.
The current Jetson offload mode uses direct Roboflow model inference rather than a Roboflow workflow.

## Invariants

- eNodeB remains radio-only.
- EPC remains control and orchestration hub.
- Car does not decide global policy; it executes received commands.
- No firmware upgrades in project operations.
- The web runtime is the operator surface; no separate GUI scripts are part of the normal flow.
