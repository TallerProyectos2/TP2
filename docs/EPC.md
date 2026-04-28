# EPC Machine Specification

## Scope

Machine-specific source of truth for the EPC host under the current script-first operating model.

## Baseline Identity

- Hostname: `tp2-EPC`
- Validation baseline updated: `2026-03-10`

## Network Bindings

- Upstream interface:
  - `eno1`
  - `10.0.128.174/24` (DHCP at last validation)
- eNodeB backhaul:
  - `enp1s0`
  - `10.10.10.1/24`
- SGi side:
  - `srs_spgw_sgi`
  - `172.16.0.1/24`
- Tailscale:
  - `tailscale0`
  - `100.97.19.112/32`

## LTE Core Contract

- Runtime process: `srsepc`
- Primary config: `/home/tp2/.config/srsran/epc.conf`
- HSS DB: `/home/tp2/.config/srsran/user_db.csv`
- Core listeners:
  - `10.10.10.1:36412/SCTP` (S1-MME)
  - `10.10.10.1:2152/UDP` (GTP-U)

## UE Provisioning Contract

- UE pool network: `172.16.0.0/24`
- Previous fixed car mapping target:
  - IMSI `901650000052126` -> `172.16.0.2`
- Live assignment note (`2026-04-27`):
  - `/home/tp2/.config/srsran/user_db.csv` currently shows `IP_alloc=dynamic`
  - latest observed assignment in `srsepc` log: `IMSI: 901650000052126, UE IP: 172.16.0.4`

## Routing And DNS Contract

- `net.ipv4.ip_forward=1`
- Persistent NAT/bootstrap script:
  - `/usr/local/sbin/tp2-ue-routing.sh`
- Startup unit:
  - `tp2-ue-routing.service`
- NAT rule:
  - `POSTROUTING -s 172.16.0.0/24 -o eno1 -j MASQUERADE`
- Optional UE DNS:
  - `dnsmasq` on `172.16.0.1:53`

## Script Runtime Contract On EPC

Operational files are stored in repo `servicios/` and validated on EPC under `/home/tp2/TP2_red4G/servicios/`.

- Car web/control runtime:
  - `coche.py` (`172.16.0.1:20001` UDP, `0.0.0.0:8088` web)
  - `autonomous_driver.py` deterministic autonomous controller used by `coche.py`
  - optional session recorder under `TP2_SESSION_RECORD_DIR` for dataset candidates
- Inference files:
  - `start_local_inference_server.py` (optional EPC local endpoint, default `127.0.0.1:9001`)
  - `inferencia.py` (CLI execution and annotated output)
  - `roboflow_runtime.py` (shared Roboflow client/helpers for CLI and live car control)
  - current remote inference profile for `coche.py`: `/home/tp2/.config/tp2/coche-jetson.env` on EPC
  - current Jetson Roboflow target: direct model inference with `ROBOFLOW_MODEL_ID=tp2-g4-2026/2`
- Conda runtime:
  - env name: `tp2`
  - activation hook: `/home/tp2/miniforge3/envs/tp2/etc/conda/activate.d/tp2-runtime.sh`
  - shell init: `conda init` applied for `zsh` and `bash`
  - operator command: `conda activate tp2`

## Service Port Contract (Current)

- `36412/SCTP`: `srsepc` S1-MME
- `2152/UDP`: `srsepc` GTP-U
- `53/TCP,UDP`: optional `dnsmasq`
- `20001/UDP`: car control runtime (`coche.py`)
- `8088/TCP`: live `coche.py` operator web view with camera, browser control, and inference status
  - `POST /mode`: switch between `manual` and `autonomous`
  - `POST /recording`: start/stop dataset capture
  - `GET /recording.json`: inspect current recording state
- `9001/TCP`: local inference endpoint (when started)
- Remote inference offload last validated from EPC to Jetson: `100.115.99.8:9001`

## Storage And Log Paths

- `/srv/tp2/frames`
  - default base for autonomous dataset captures: `/srv/tp2/frames/autonomous`
- `/srv/tp2/logs`
- `/srv/tp2/docker`
- `/srv/tp2/config`
- LTE logs:
  - `/srv/tp2/logs/srsepc.log`
  - `/srv/tp2/logs/srsepc-console.log` (when launched with console redirect)

## Evidence References

- EPC phase baselines:
  - `docs/logs/validations/2026-03-04-epc-fase-0.md`
  - `docs/logs/validations/2026-03-04-epc-fase-1.md`
  - `docs/logs/validations/2026-03-05-epc-fase-2.md`
- EPC local inference:
  - `docs/logs/validations/2026-03-05-epc-inferencia-local.md`
- Car UE assignment:
  - `docs/logs/validations/2026-03-10-car-ue-ip-assignment.md`
