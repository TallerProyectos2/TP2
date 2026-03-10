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
- Current fixed car mapping:
  - IMSI `901650000052126` -> `172.16.0.2`
- Last observed assignment in `srsepc` log:
  - `IMSI: 901650000052126, UE IP: 172.16.0.2`

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

Operational scripts are stored in repo `servicios/` and validated on EPC under `/home/tp2/servicios_tp2/`.

- Car control scripts:
  - `car1_cloud_control_server.py`
  - `car1_cloud_control_server_real_time_control.py`
  - `car1_manual_control_server.py`
  - `car3_cloud_control_server.py`
  - `car3_cloud_control_server_real_time_control.py`
  - `car3_manual_control_server.py`
  - shared logic: `artemis_autonomous_car.py`
- Inference scripts:
  - `start_local_inference_server.py` (local endpoint, default `127.0.0.1:9001`)
  - `inferencia.py` (CLI execution and annotated output)
  - `inferencia_gui_web.py` (web UI, default `0.0.0.0:7860`)
  - `inferencia_gui.py` (desktop GUI; requires `tkinter`)

## Service Port Contract (Current)

- `36412/SCTP`: `srsepc` S1-MME
- `2152/UDP`: `srsepc` GTP-U
- `53/TCP,UDP`: optional `dnsmasq`
- `20001/UDP`: car1 control scripts (as currently hardcoded)
- `20003/UDP`: car3 control scripts (as currently hardcoded)
- `9001/TCP`: local inference endpoint (when started)
- `7860/TCP` or `7861/TCP`: inference GUI web (depending on launch args)

## Storage And Log Paths

- `/srv/tp2/frames`
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
