# TP2 Network Contract

## Backhaul

- Network: `10.10.10.0/24`
- EPC: `10.10.10.1`
- eNodeB: `10.10.10.2`

This link carries S1 control and user-plane traffic between `srsepc` and `srsenb`.

## EPC Interfaces

- `enp1s0` (backhaul): `10.10.10.1`
- `srs_spgw_sgi` (UE side): `172.16.0.1`
- `eno1` (upstream): NAT egress interface
- `tailscale0`: operator remote access

## UE Routing Path

- UE subnet: `172.16.0.0/24`
- Forwarding enabled on EPC
- NAT/MASQUERADE from `172.16.0.0/24` to `eno1`
- Optional DNS for UE on `172.16.0.1:53`

## UE Addressing

- Car IMSI: `901650000052126`
- Current fixed assignment: `172.16.0.2`

## Current Runtime Reachability

- Car must reach EPC control script UDP endpoint.
- EPC must return UDP control packets to car.
- EPC local inference endpoint is usually loopback (`127.0.0.1:9001`), not external by default.

## Jetson Reachability (Pending Integration)

- Jetson must be reachable from EPC only.
- Car should not call Jetson directly.
- When Jetson inference is enabled, keep EPC local inference as fallback.

## Core Ports

- `36412/SCTP`: EPC S1-MME
- `2152/UDP`: EPC GTP-U
- `53/TCP,UDP`: EPC DNS (optional)
- `20001/UDP`: car1 control script endpoint
- `20003/UDP`: car3 control script endpoint
- `9001/TCP`: EPC local inference endpoint
- `7860/TCP` or `7861/TCP`: inference GUI web (when launched)

## Validation Checklist

- EPC can ping eNodeB
- eNodeB has S1 association to EPC
- Car UE is attached with `172.16.0.2`
- EPC control script receives UDP payloads from car
- EPC sends UDP control back to car
- If local inference is enabled, `127.0.0.1:9001` responds
