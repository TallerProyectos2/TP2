# 2026-05-07 - BMI160 IMU speed control EPC deployment

## Scope

- Deployed the BMI160-aware EPC runtime to `/home/tp2/TP2_red4G` on `tp2-EPC`.
- Restarted `tp2-car-control.service` so the active runtime uses the updated `servicios/coche.py`.
- Kept the car-side SoftARTEMIS update as repository content only; the operator will load it on the car.

## Pre-Deployment State

- EPC host: `tp2-EPC`.
- EPC repo: `/home/tp2/TP2_red4G`.
- Branch: `main...origin/main`.
- Existing unrelated remote worktree state preserved: `D servicios/test.jpg`.
- Service before deployment: `tp2-car-control.service` active.

## Commands

```bash
rsync -avR ARCHITECTURE.md MACHINES.md PLAN.md RUNBOOK.md SoftARTEMIS/cloud_control_node_UDP.py docs/CAR-AGENT.md docs/EPC.md servicios/coche.py tests/test_coche_runtime.py docs/logs/validations/2026-05-07-bmi160-imu-speed-control-local.md tp2@100.97.19.112:/home/tp2/TP2_red4G/
ssh tp2@100.97.19.112 'cd /home/tp2/TP2_red4G && /home/tp2/miniforge3/bin/conda run --no-capture-output -n tp2 python -m py_compile servicios/coche.py servicios/autonomous_driver.py servicios/lidar_processor.py'
ssh tp2@100.97.19.112 'cd /home/tp2/TP2_red4G && PYTHONPATH=servicios /home/tp2/miniforge3/bin/conda run --no-capture-output -n tp2 python -m unittest discover -s tests -p "test_coche_runtime.py"'
ssh tp2@100.97.19.112 'cd /home/tp2/TP2_red4G && PYTHONPATH=servicios /home/tp2/miniforge3/bin/conda run --no-capture-output -n tp2 python -m unittest discover -s tests'
```

## Results

- EPC `py_compile`: OK.
- EPC `test_coche_runtime.py`: 26 tests OK.
- EPC full test suite: 67 tests OK.
- `sudo systemctl restart tp2-car-control.service` was unavailable because sudo required an interactive password.
- Because the service process runs as `tp2` and the unit has `Restart=on-failure`, the runtime was restarted by killing the old `tp2` process group and letting systemd relaunch it.
- Service after restart: active since `2026-05-07 12:48:52 CEST`.
- Active runtime PIDs after restart: `394803` (`conda`) and `394808` (`python -u /home/tp2/TP2_red4G/servicios/coche.py`).
- UDP listener after restart: `172.16.0.1:20001`.
- Web listener after restart: `0.0.0.0:8088`.

## Runtime Status

Status endpoint after restart:

```json
{
  "control": {
    "armed": false,
    "mode": "manual",
    "seq": 1,
    "source": "neutral"
  },
  "imu": {
    "enabled": true,
    "errors": 0,
    "frames": 218,
    "last_error": null,
    "speed_control_enabled": true,
    "status": "ready"
  },
  "inference": "running",
  "lidar": {
    "enabled": true,
    "errors": 0,
    "frames": 45,
    "status": "clear"
  },
  "udp": {
    "bind": "172.16.0.1:20001",
    "last_client": "172.16.0.11:48467",
    "last_packet_age_sec": 0.024,
    "last_packet_type": "I",
    "tx_packets": 1400
  }
}
```

Additional IMU sample:

```json
{
  "accel_mps2": {
    "x": 0.0957,
    "y": 0.1029,
    "z": -9.7797
  },
  "estimated_speed_mps": 0.0,
  "frames": 368,
  "gyro_dps": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0
  },
  "seq": 1959,
  "speed_control": {
    "active": false,
    "reason": "manual-mode",
    "target_speed_mps": 0.0,
    "throttle_correction": 0.0
  },
  "status": "ready"
}
```

## Notes

- No synthetic UDP `D` packet was injected into the active EPC runtime because the real car client `172.16.0.11:48467` was already connected. Injecting from the EPC would temporarily replace `last_client` and could interrupt command TX to the car.
- IMU PID speed assistance is expected to remain inactive while the runtime is in manual mode. It becomes active only in autonomous forward actions with fresh IMU telemetry.
