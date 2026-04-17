# MacBook Launcher S1 And UE Wait Hardening

- Date: `2026-04-17`
- Operator host: MacBook workspace running `ops/`

## Goal

Harden the startup tooling so it does not continue past LTE startup until:

- `bladeRF-cli` has released the SDR after FPGA load
- EPC and eNodeB have a real SCTP S1 association
- the car UE is required by default in the standard MacBook launcher config

## Runtime Inspection

Read-only checks on the live lab during this task showed:

- EPC at `2026-04-17 09:41:20 CEST`:
  - `srsepc` active as `/usr/local/bin/srsepc /home/tp2/.config/srsran/epc.conf`
  - SCTP listener present on `10.10.10.1:36412`
  - UDP GTP-U listener present on `10.10.10.1:2152`
  - recent `journalctl -u tp2-srsepc.service` contains `Received S1 Setup Request` and `Sending S1 Setup Response`
- EPC SCTP state at `2026-04-17 09:41:20 CEST`:
  - established association observed:
    - `10.10.10.1:36412 <-> 10.10.10.2:38119`
- eNodeB at `2026-04-17 10:41:52` according to its local clock:
  - `srsenb` active as `/usr/local/bin/srsenb /home/tp2/.config/srsran/enb.conf`
  - no `bladeRF-cli` or `bladerf-cli` process present
  - no USB file holder for the SDR found via `lsof`
  - `bladeRF` detected as `Nuand LLC bladeRF 2.0 micro`
  - `journalctl -u tp2-bladerf-fpga.service` shows successful FPGA load immediately before `srsenb` startup
  - `journalctl -u tp2-srsenb.service` shows `==== eNodeB started ===`
- Current remaining gap:
  - `ops/bin/tp2-status` still reports `car UE: not confirmed`
- Additional radio signal from live `enb.log`:
  - repeated `RF Overflow` messages were present in the recent log window

## Changes

- `ops/lib/tp2-common.sh`
  - added `tp2_wait_bladerf_cli_released`
  - added `tp2_wait_s1`
  - changed default `TP2_REQUIRE_CAR_UE` to `1`
- `ops/bin/tp2-up`
  - waits for `bladeRF-cli` release after FPGA load
  - waits for real S1 association after `srsenb` starts
- `ops/bin/tp2-status`
  - reports `S1 association: established|missing`
  - reports `bladeRF-cli: released|busy`
- `ops/bin/tp2-validate`
  - validates `bladeRF-cli` release when `srsenb` is active
  - validates S1 association when both EPC and eNodeB are active
- `ops/tp2-lab.env.example`
  - now defaults `TP2_REQUIRE_CAR_UE=1`

## Validation

- `bash -n` passed for:
  - `ops/lib/tp2-common.sh`
  - `ops/bin/tp2-up`
  - `ops/bin/tp2-status`
  - `ops/bin/tp2-validate`
  - `ops/bin/tp2-enb-load-fpga`
- `ops/bin/tp2-status` from the MacBook reported:
  - `S1 association: established`
  - `bladeRF-cli: released`
  - `car UE: not confirmed`
- `ops/bin/tp2-validate` from the MacBook completed with:
  - `bladeRF-cli release check: ok`
  - `S1 association check: ok`

## Result

- The launcher now blocks on the actual LTE control-plane milestones instead of open ports alone.
- The live issue at the end of this task is not a retained `bladeRF-cli` process and not a missing EPC-eNodeB S1 association.
- The remaining live problem is that the UE still does not attach during the observed window.
