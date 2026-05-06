# LiDAR Top-Down Tuning EPC Deploy

## Scope

- Target machine: `tp2-EPC`
- Runtime service: `tp2-car-control.service`
- Change: top-down LiDAR web view, runtime LiDAR/autonomous tuning controls, and 45-point frontal LiDAR collision sector.
- No LTE, eNodeB, Jetson, car-side, or firmware changes were made.

## Local Validation

```text
python -m py_compile servicios/lidar_processor.py servicios/coche.py
OK

PYTHONPATH=servicios python -m unittest discover -s tests
Ran 64 tests, OK
```

Covered checks include:

- flat Artemis/professor LiDAR range lists keep beam indices and sample count
- 45 frontal beams at `0.15 m` trigger `lidar-stop`
- a beam just outside the 45-point frontal sector does not trigger a frontal stop
- LiDAR stop overrides autonomous forward motion and autonomous safe fallback
- `/settings` accepts LiDAR runtime tuning values

## EPC Deployment

```text
cd /home/tp2/TP2_red4G
git pull --ff-only origin main
/home/tp2/miniforge3/bin/conda run --no-capture-output -n tp2 python -m py_compile servicios/lidar_processor.py servicios/coche.py
systemctl restart tp2-car-control.service
```

Live service state after restart:

```text
tp2-car-control.service active
ExecMainPID=293127
172.16.0.1:20001/UDP listening
0.0.0.0:8088/TCP listening
```

Live `/status.json` after deployment:

```json
{
  "service_lidar_status": "clear",
  "frames": 122,
  "point_count": 246,
  "front_point_count": 45,
  "front_angle_deg": 22.5,
  "stop_distance_m": 0.15,
  "settings_lidar_stop": 0.15,
  "settings_lidar_front_points": 45,
  "settings_cruise": 0.55,
  "error": null
}
```

Live HTML smoke:

```text
LiDAR seguridad
data-setting="lidar_stop_distance_m"
LIDAR TOP
```

## Result

The EPC is serving the new runtime. The LiDAR stage now renders a top-down reconstruction and `/settings` exposes LiDAR safety and autonomous driving parameters. The deployed default collision behavior uses 45 frontal LiDAR beams and stops autonomous control at or below `0.15 m`.
