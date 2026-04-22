# TP2 Machine Inventory

## Purpose

Record current machine roles, addressing, and access paths for the live lab.

Do not store passwords here.

## PC EPC

- Role:
  - LTE core host
  - main script runtime host
- Hostname:
  - `tp2-EPC`
- Services:
  - `srsepc`
  - NAT / IP forwarding
  - optional `dnsmasq`
  - control scripts from `servicios/`
  - local inference endpoint from `servicios/`
- Known addresses:
  - upstream (`eno1`): `10.0.128.174/24` (DHCP)
  - backhaul (`enp1s0`): `10.10.10.1/24`
  - Tailscale: `100.97.19.112`
  - SGi (`srs_spgw_sgi`): `172.16.0.1/24`
- Script runtime references:
  - repository path: `servicios/`
  - deployed path used in recent validations: `/home/tp2/TP2_red4G/servicios/`
- Access path:
  - primary remote entrypoint from the operator MacBook
  - direct Tailscale SSH: `ssh tp2@100.97.19.112`
- Operator automation path:
  - run `ops/bin/tp2-*` from `/home/tp2/TP2_red4G` on the EPC
  - EPC-targeted operations use `TP2_EPC_SSH=local`
- Autostart references:
  - `ops/systemd/epc/tp2-srsepc.service`
  - `ops/systemd/epc/tp2-local-inference.service`
  - `ops/systemd/epc/tp2-car-control.service`
  - `ops/systemd/epc/tp2-car-command-am-cloud.service`
  - `ops/sudoers/epc/tp2-lab` for narrow passwordless TP2 service control
  - live operator view: `http://100.97.19.112:8088/`

## PC eNodeB

- Role:
  - LTE radio access
- Hostname:
  - `tp2-ENB`
- Services:
  - `srsenb`
  - `bladeRF`
- Known addresses:
  - EPC-side SSH/access IP: `10.10.10.2`
  - backhaul: `10.10.10.2`
  - Tailscale: `100.69.186.34`
- Access path:
  - operator path from a MacBook: `ssh -J tp2@100.97.19.112 tp2@10.10.10.2`
  - fallback operator path: SSH to `tp2@100.97.19.112` on EPC, then SSH from EPC to `tp2@10.10.10.2`
  - `tp2@EPC -> tp2@eNodeB` key-based SSH hop validated (`2026-03-04`)
- Operator automation path:
  - from EPC, use `TP2_ENB_SSH=tp2@10.10.10.2`
- Autostart references:
  - `ops/systemd/enb/tp2-enb-link.service` runs `/home/tp2/to_epc_link.sh` at eNodeB boot
  - `ops/systemd/enb/tp2-bladerf-fpga.service`
  - `ops/systemd/enb/tp2-srsenb.service`
  - `ops/sudoers/enb/tp2-lab` for narrow passwordless TP2 service control

## Jetson

- Role:
  - inference-only node
- Hostname:
  - `tp2-jetson`
- SSH user:
  - `grupo4`
- Target use:
  - offload inference from EPC when required
  - keep control path anchored on EPC
  - last validated HTTP endpoint from EPC: `http://100.115.99.8:9001`
- Runbook:
  - `docs/JETSON.md`
- Addressing:
  - management LAN IP: `192.168.72.127`
  - Tailscale IP: `100.115.99.8`
  - must be reachable from EPC
- Access path:
  - primary operator path from a MacBook: `ssh grupo4@100.115.99.8`
  - direct SSH by hostname when available: `ssh grupo4@tp2-jetson`
  - management LAN SSH: `ssh grupo4@192.168.72.127`
- Autostart references:
  - `ops/systemd/jetson/tp2-roboflow-inference.service`

## Coche

- Role:
  - mobile client
- Services:
  - sends sensor payloads to EPC control scripts
  - executes steering/throttle commands from EPC
- Connectivity:
  - attaches as LTE UE
  - receives address from EPC UE pool `172.16.0.0/24`
  - observed IMSI: `901650000052126`
  - static UE assignment on EPC: `901650000052126 -> 172.16.0.2`
- Control transport:
  - UDP payload stream to EPC control scripts
  - UDP control packets returned by EPC
- Autostart note:
  - car-side runtime is started manually by operators
  - EPC automation checks UE IP `172.16.0.2` best-effort before publishing the `AM-Cloud` state, but does not block by default

## Ownership Rules

- EPC owns orchestration and control runtime.
- eNodeB owns radio only.
- Jetson owns inference only when explicitly selected by EPC configuration.
- Car owns capture/sensor emission and command execution only.
