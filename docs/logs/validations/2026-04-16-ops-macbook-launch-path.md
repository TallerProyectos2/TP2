# TP2 Ops MacBook Launch Path Update

- Date: `2026-04-16`
- Scope: local repo `ops/` recovery and startup-path correction

## Goal

Track the startup tooling in this repository and correct two operational issues:

- the launcher was EPC-centric, so Jetson startup depended on EPC-to-Jetson SSH;
- the FPGA preload step could be skipped by stale `active (exited)` systemd state on the eNodeB.

## Evidence Used

- Live EPC repo copy under `/home/tp2/TP2_red4G/ops/` was inspected read-only to recover the current startup scripts and units.
- Live eNodeB inspection showed:
  - `tp2-bladerf-fpga.service` last succeeded on `2026-04-14 11:49:44 CEST`;
  - `tp2-srsenb.service` failed on `2026-04-16 10:33:17` through `10:33:28 CEST`;
  - failure strings included `FPGA bitstream file not found` and `current "Firmware Loaded", requires "Initialized"`.
- Live Jetson inspection showed on `2026-04-16`:
  - `tp2-roboflow-inference.service` was already `active`;
  - local endpoint check `http://127.0.0.1:9001/openapi.json` succeeded;
  - `grupo4` direct SSH worked, but `sudo -n` still required a password for `systemctl start`.

## Repo Changes

- Added tracked `ops/` scripts and systemd templates to this repo.
- Changed launcher defaults to:
  - direct MacBook -> EPC SSH
  - direct MacBook -> Jetson SSH
  - MacBook -> eNodeB through EPC proxy
- Updated `ops/bin/tp2-up` to stop/start the FPGA oneshot service before eNodeB start when `srsenb` is not already active.
- Updated the eNodeB `tp2-srsenb.service` template so actual `srsenb` starts reload the FPGA bitstream through `ExecStartPre`.
- Updated `ops/bin/tp2-up` to skip Jetson `systemctl start` when the service is already active and to warn clearly when Jetson start would require passworded sudo.
- Added the missing `ops/bin/tp2-install-systemd`, `ops/bin/tp2-install-sudoers`, and `ops/sudoers/` files so MacBook-driven deployment is tracked in-repo too.

## Verification Performed

- `bash -n` planned for all new `ops/bin/*` scripts in this repo checkout.
- Documentation updated to reflect the new operator access path.

## Verification Not Yet Performed

- No full runtime deployment of the new `ops/` tree to EPC/eNodeB/Jetson was performed in this task.
- No end-to-end LTE attach or car control validation was executed from the new MacBook-driven launcher yet.

## Required Next Runtime Check

1. Deploy the updated `ops/` files to the live machines.
2. Run `ops/bin/tp2-up jetson` from a MacBook with Tailscale access.
3. Confirm:
   - `tp2-bladerf-fpga.service` reruns on this session start
   - `tp2-srsenb.service` remains active
   - Jetson inference starts through direct MacBook SSH
   - EPC can still reach the Jetson inference endpoint
   - LTE sockets and UE path validate normally
