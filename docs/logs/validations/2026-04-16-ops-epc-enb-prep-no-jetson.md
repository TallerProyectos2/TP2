# TP2 Ops Prep On EPC And eNodeB With Jetson/BladeRF Disconnected

- Date: `2026-04-16`
- Scope: MacBook-driven TP2 ops preparation for next lab session

## Constraints During This Task

- Jetson was disconnected and not reachable by SSH.
- `bladeRF` was physically disconnected from the eNodeB.
- Because of that, no end-to-end LTE or Jetson inference startup validation was attempted.

## Read-Only Findings Before Changes

- EPC:
  - `/etc/sudoers.d/tp2-lab` did not exist.
  - `tp2-srsepc.service`, `tp2-local-inference.service`, and `tp2-car-control.service` were already installed.
- eNodeB:
  - `/etc/sudoers.d/tp2-lab` did not exist.
  - `tp2-srsenb.service` still used the older dependency model:
    - `After=... tp2-bladerf-fpga.service`
    - `Requires=tp2-bladerf-fpga.service`
    - no `ExecStartPre=/usr/local/sbin/tp2-enb-load-fpga ...`
  - `tp2-bladerf-fpga.service` showed stale `active` state even with the hardware removed.

## Changes Applied

- EPC:
  - installed `/etc/sudoers.d/tp2-lab` from `ops/sudoers/epc/tp2-lab`
- eNodeB:
  - installed `/etc/sudoers.d/tp2-lab` from `ops/sudoers/enb/tp2-lab`
  - installed updated `/usr/local/sbin/tp2-enb-load-fpga`
  - installed updated units:
    - `/etc/systemd/system/tp2-enb-link.service`
    - `/etc/systemd/system/tp2-bladerf-fpga.service`
    - `/etc/systemd/system/tp2-srsenb.service`
  - ran `systemctl daemon-reload`
  - stopped `tp2-bladerf-fpga.service` so the stale oneshot state is cleared before next session

## Verification Evidence

- EPC:
  - `sudo -n -l /usr/bin/systemctl start tp2-srsepc.service` succeeded
  - `sudo -n -l /usr/bin/install -m 0644 /tmp/tp2-srsepc.service /etc/systemd/system/tp2-srsepc.service` succeeded
- eNodeB:
  - `sudo -n -l /usr/bin/systemctl start tp2-srsenb.service` succeeded
  - `sudo -n -l /usr/bin/install -m 0755 /tmp/tp2-enb-load-fpga /usr/local/sbin/tp2-enb-load-fpga` succeeded
  - `systemctl is-active tp2-bladerf-fpga.service` returned `inactive`
  - installed `tp2-srsenb.service` now contains:
    - `ExecStartPre=/usr/local/sbin/tp2-enb-load-fpga /home/tp2/Descargas/hostedxA9.rbf`

## Remaining Blockers

- Jetson sudoers and unit deployment could not be applied because Jetson was offline.
- No FPGA load could be tested because the `bladeRF` was disconnected.
- No `tp2-up jetson` runtime validation was performed because both blocked components are required for full path validation.

## Recommended Next Session Sequence

1. Reconnect Jetson and confirm direct SSH from the MacBook.
2. Install `ops/sudoers/jetson/tp2-lab` on Jetson.
3. Reconnect `bladeRF` to the eNodeB.
4. Run `ops/bin/tp2-status`.
5. Run `ops/bin/tp2-up jetson`.
6. Confirm:
   - `tp2-bladerf-fpga.service` reruns successfully
   - `tp2-srsenb.service` stays active
   - Jetson inference is reachable from EPC
   - LTE and UE path checks complete
