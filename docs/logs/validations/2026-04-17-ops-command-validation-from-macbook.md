# MacBook Startup Command Validation

- Date: `2026-04-17`
- Operator host: MacBook workspace running `ops/` from the repo root

## Goal

Confirm that the MacBook launcher and status tooling validate the live EPC and eNodeB runtime against the expected command lines:

- EPC: `/usr/local/bin/srsepc /home/tp2/.config/srsran/epc.conf`
- eNodeB: `/usr/local/bin/srsenb /home/tp2/.config/srsran/enb.conf`

## Changes

- Updated `ops/bin/tp2-up` to wait for the exact live `srsepc` and `srsenb` command lines after starting the services.
- Updated `ops/bin/tp2-status` to print the matching live process lines from EPC and eNodeB and to report current car UE visibility.
- Updated `ops/bin/tp2-validate` to assert the expected live command lines when the LTE services are active.
- Corrected EPC LTE socket checks to use protocol-specific `ss` queries:
  - SCTP `36412` via `ss -H -lnA sctp`
  - UDP `2152` via `ss -H -lun`

## Runtime Evidence

- `ops/bin/tp2-status` from the MacBook reported:
  - EPC:
    - `srsepc: active`
    - `srsepc cmd: 329274 /usr/local/bin/srsepc /home/tp2/.config/srsran/epc.conf`
    - S1-MME listener present on `10.10.10.1:36412`
    - GTP-U listener present on `10.10.10.1:2152`
    - car-control UDP listener present on `172.16.0.1:20001`
    - live web endpoint present on `0.0.0.0:8088`
    - `car UE: not confirmed`
  - eNodeB:
    - `srsenb: active`
    - `srsenb cmd: 25074 /usr/local/bin/srsenb /home/tp2/.config/srsran/enb.conf`
    - `bladeRF` detected as `Nuand LLC bladeRF 2.0 micro`
  - Jetson:
    - inference service active
    - `http://127.0.0.1:9001/openapi.json` reachable
- `ops/bin/tp2-validate` from the MacBook completed successfully with:
  - `EPC srsepc command check: ok`
  - `eNodeB srsenb command check: ok`
  - `LTE socket check: ok or EPC core inactive`

## Result

- The startup tooling now proves from the MacBook that the live EPC and eNodeB processes match the intended `srsepc` and `srsenb` command lines.
- The current remaining runtime gap is not the launch command path: the car UE was still not confirmed at validation time.
