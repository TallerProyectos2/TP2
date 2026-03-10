# Car Runtime Contract (Current)

## Purpose

Document the current car-side interaction model used by existing EPC scripts.

## Control Model

The active operational model is UDP stream + UDP control:

- Car sends payloads to EPC control server.
- EPC script processes payloads and returns control command.

No new backend API is required to run this path.

## Script Endpoints In Use

Control servers available in `servicios/`:

- `car1_cloud_control_server.py`
- `car1_cloud_control_server_real_time_control.py`
- `car1_manual_control_server.py`
- `car3_cloud_control_server.py`
- `car3_cloud_control_server_real_time_control.py`
- `car3_manual_control_server.py`

Shared control logic:

- `artemis_autonomous_car.py`

## UDP Packet Contract (As Implemented)

- Incoming payload discriminator (first byte):
  - `I`: camera image payload
  - `L`: lidar payload
  - `B`: battery level
  - `D`: reserved/other data path
- Payload body is deserialized with `pickle.loads(...)` in current scripts.
- Outgoing control packet type:
  - `C` + steering (`double`) + throttle (`double`)

## Runtime Modes

- Manual mode:
  - keyboard-driven steering/throttle
  - camera/LIDAR display for operator feedback
- Autonomous mode:
  - image/LIDAR processed by `artemis_autonomous_car`
  - steering/throttle computed automatically
- Real-time autonomous mode:
  - keyboard switches route/behavior mode while autonomous loop runs

## LTE Binding Context

- Car attaches as UE in EPC network.
- Current static mapping:
  - IMSI `901650000052126` -> `172.16.0.2`

## Minimum Validation

1. Start LTE (`srsepc` + `srsenb`) and verify UE attach.
2. Start chosen EPC control script.
3. Confirm script receives UDP payloads from car.
4. Confirm control packets are returned and car responds.
