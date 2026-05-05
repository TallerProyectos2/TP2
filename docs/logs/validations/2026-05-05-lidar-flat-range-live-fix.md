# LiDAR Flat Range Live Fix

## Scope

- Target machine: `tp2-EPC`
- Runtime service: `tp2-car-control.service`
- Issue: web UI showed LiDAR `empty` while the LiDAR was connected.
- No LTE, eNodeB, Jetson, or firmware changes were made.

## Findings

`/status.json` showed that packets were arriving:

```text
udp.packets.L = 2634
lidar.frames = 2634
lidar.errors = 0
lidar.point_count = 0
lidar.status = empty
```

Initial `tcpdump` attempt without sudo failed because interactive sudo was required. After operator approval, a short `tcpdump` capture on EPC confirmed live `L` packets:

```text
srs_spgw_sgi In IP 172.16.0.2.36167 > 172.16.0.1.20001: UDP, length 5827
payload starts with: L(Finf...
3 packets captured
0 packets dropped by kernel
```

The payload format matches the professor/Artemis scripts: a pickled plain list of LiDAR ranges, with `inf` values for no-return samples. The parser only handled dict-based `ranges` or Cartesian point lists, so a flat list of floats was treated as invalid point records and became empty.

## Fix

- Updated `servicios/lidar_processor.py` to treat a flat numeric list as a 360-degree range scan.
- Preserved original indices when entries are `inf`, so finite measurements keep the correct angle.
- Added tests for flat professor ranges and `inf` index preservation.

## Validation

Local:

```text
PYTHONPATH=servicios python -m unittest discover -s tests -p 'test_lidar_processor.py'
Ran 7 tests, OK

PYTHONPATH=servicios python -m unittest discover -s tests
Ran 61 tests, OK
```

EPC:

```text
git pull --ff-only origin main
python -m py_compile servicios/lidar_processor.py servicios/coche.py
systemctl restart tp2-car-control.service
```

Live `/status.json` after restart:

```text
udp.packets.L = 101
lidar.frames = 101
lidar.point_count = 274
lidar.points length = 274
lidar.safety.status = clear
lidar.safety.min_front_distance_m = 2.318
lidar.error = null
```

The web LiDAR view should now render points instead of `empty`.
