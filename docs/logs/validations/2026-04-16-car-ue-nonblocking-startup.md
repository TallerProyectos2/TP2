# 2026-04-16 - Car UE non-blocking startup check

## Scope

- Removed the blocking `120s` car UE wait from the default `tp2-up` path.
- Added `TP2_REQUIRE_CAR_UE=0` as the default startup behavior.
- Kept the old strict behavior available by setting `TP2_REQUIRE_CAR_UE=1`.
- Updated startup docs to clarify that `172.16.0.2` is checked best-effort before publishing `AM-Cloud`.
- No radio services, control services, or firmware actions were started during validation.

## Files Changed

- `ops/tp2-lab.env.example`
- `ops/lib/tp2-common.sh`
- `ops/bin/tp2-up`
- `docs/AUTOSTART.md`
- `RUNBOOK.md`
- `MACHINES.md`

## Validation

Local syntax check:

```bash
bash -n ops/lib/tp2-common.sh ops/bin/tp2-up
```

Expected runtime behavior:

- If `172.16.0.2` is visible, `tp2-up` logs the car UE check as `ok`.
- If `172.16.0.2` is not visible, `tp2-up` logs a warning and continues:
  - `not confirmed; continuing because TP2_REQUIRE_CAR_UE=0`
- To restore the old blocking behavior, set:
  - `TP2_REQUIRE_CAR_UE=1`

## Deployment

The change was synced to the EPC checkout at `/home/tp2/TP2_red4G`.

Remote checks on EPC:

- `ops/tp2-lab.env.example` contains `TP2_REQUIRE_CAR_UE=0`.
- `ops/bin/tp2-up` calls `tp2_maybe_wait_car_ue` instead of the blocking wait.
- No override for `TP2_REQUIRE_CAR_UE` was present in:
  - `/etc/tp2/lab.env`
  - `/home/tp2/.config/tp2/lab.env`
  - `/home/tp2/TP2_red4G/ops/tp2-lab.env`
- Remote syntax check passed:
  - `bash -n ops/lib/tp2-common.sh ops/bin/tp2-up`
- Non-startup function check completed in 1 second:
  - `tp2_maybe_wait_car_ue "Testing car UE nonblocking"`
