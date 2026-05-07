# 2026-05-07 - BMI160 IMU speed control local validation

## Scope

- Added BMI160 telemetry emission to `SoftARTEMIS/cloud_control_node_UDP.py` as UDP `D` JSON using schema `tp2.car.telemetry.v1`.
- Added EPC-side JSON `D` parsing, IMU status reporting, and a bounded PID speed-control layer in `servicios/coche.py`.
- Camera/Roboflow decisions, lane assist, and LiDAR safety remain the autonomous driving inputs; IMU only adjusts forward throttle when fresh and valid.

## Local Validation

```bash
python3 -m py_compile servicios/coche.py servicios/autonomous_driver.py servicios/lidar_processor.py
python3 -m py_compile SoftARTEMIS/cloud_control_node_UDP.py
PYTHONPATH=servicios python3 -m unittest discover -s tests -p 'test_coche_runtime.py'
PYTHONPATH=servicios python3 -m unittest discover -s tests -p 'test_lidar_processor.py'
PYTHONPATH=servicios python3 -m unittest discover -s tests
TP2_ENABLE_INFERENCE=0 TP2_BIND_IP=127.0.0.1 TP2_BIND_PORT=29001 TP2_WEB_HOST=127.0.0.1 TP2_WEB_PORT=18088 TP2_SESSION_RECORD_AUTOSTART=0 PYTHONPATH=servicios python3 servicios/coche.py
```

Results:

- `py_compile`: OK.
- `SoftARTEMIS/cloud_control_node_UDP.py` syntax compile under Python 3 parser: OK.
- `test_coche_runtime.py`: 26 tests OK.
- `test_lidar_processor.py`: 9 tests OK.
- Full local test suite: 67 tests OK.
- Local UDP/HTTP smoke: synthetic `D + JSON` IMU packet returned a `C` control packet and `/status.json` reported `imu.status=ready`, `imu.frames=2`, `imu.seq=100`, `imu.received_age_sec=0.013`.

## Notes

- The local workstation does not have `python2`, so the legacy SoftARTEMIS runtime was not byte-compiled locally.
- No remote EPC/car deployment was performed in this validation. Live validation still needs the operator to load the updated SoftARTEMIS on the car and confirm `/status.json` reports `imu.status=ready`, `imu.frames > 0`, and non-stale `imu.speed_control`.
