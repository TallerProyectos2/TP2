# Temporary Speed Sign Policy EPC Deployment Validation

- Date: 2026-05-15
- Host: `tp2-EPC`
- Deployed repo path: `/home/tp2/TP2_red4G`
- GitHub commit deployed: `bb402be` (`Make speed signs temporary cruise overrides`)

## Deployment

- Files copied to EPC with `rsync` from a clean worktree based on `origin/main`.
- Host-local defaults updated at `/home/tp2/.config/tp2/coche-control-defaults.json`.
- Backup created: `/home/tp2/.config/tp2/coche-control-defaults.json.bak-20260515-speed-override`.
- `tp2-car-control.service` was restarted after `systemctl restart` required interactive authentication; the running process is now a fresh instance loading the new code.

## EPC Checks

```text
PYTHONPATH=servicios python -m unittest discover -s tests
Ran 71 tests in 0.589s
OK
```

```text
systemctl is-active tp2-car-control.service
active
```

```text
ss -lunp | grep ":20001"
172.16.0.1:20001 users:(("python",pid=532608,fd=3))
```

```text
ss -lntp | grep ":8088"
0.0.0.0:8088 users:(("python",pid=532608,fd=4))
```

```text
/status.json selected fields
speed_override_sec=3.0
slow_throttle=0.5
fast_throttle=0.7
cruise_throttle=0.54
last_packet_type=None
tx_packets=0
inference_status=offline
```

## Runtime Limitation

The EPC service is active and serving the updated policy, but no live car UDP traffic was present during this validation window (`last_packet_type=None`, `tx_packets=0`). The inference status was also `offline`, so this evidence covers deployment, process/port health, and loaded runtime settings, not an end-to-end moving-car validation.
