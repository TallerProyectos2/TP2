# 2026-04-14 - Autostart orchestration scaffold

## Scope

- Added repository-owned startup orchestration to be run from the EPC for EPC, eNodeB, and Jetson.
- Car-side runtime remains operator-managed; EPC automation only checks UE IP before publishing state.
- EPC and eNodeB unit files were installed after read-only validation.
- `tp2-enb-link.service` was enabled on eNodeB so `/home/tp2/to_epc_link.sh` runs at eNodeB boot.
- No LTE radio/control session was started during this validation.
- No firmware updates were performed.

## Files Added

- `ops/tp2-lab.env.example`
- `ops/lib/tp2-common.sh`
- `ops/bin/tp2-up`
- `ops/bin/tp2-down`
- `ops/bin/tp2-status`
- `ops/bin/tp2-validate`
- `ops/bin/tp2-install-systemd`
- `ops/bin/tp2-enb-load-fpga`
- `ops/systemd/epc/tp2-srsepc.service`
- `ops/systemd/epc/tp2-local-inference.service`
- `ops/systemd/epc/tp2-car-control.service`
- `ops/systemd/epc/tp2-car-command-am-cloud.service`
- `ops/systemd/enb/tp2-enb-link.service`
- `ops/systemd/enb/tp2-bladerf-fpga.service`
- `ops/systemd/enb/tp2-srsenb.service`
- `ops/systemd/jetson/tp2-roboflow-inference.service`
- `docs/AUTOSTART.md`

## Local Validation

Commands:

```bash
bash -n ops/lib/tp2-common.sh \
  ops/bin/tp2-enb-load-fpga \
  ops/bin/tp2-up \
  ops/bin/tp2-down \
  ops/bin/tp2-status \
  ops/bin/tp2-validate \
  ops/bin/tp2-install-systemd

python3 - <<'PY'
from pathlib import Path
for path in sorted(Path('ops/systemd').rglob('*.service')):
    text = path.read_text()
    assert '[Unit]' in text and '[Service]' in text, path
    print(path)
PY
```

Result:

- Shell syntax validation passed.
- All systemd unit templates contain `[Unit]` and `[Service]` sections.

## Read-Only Remote Checks

Tailscale status from operator machine:

- EPC `100.97.19.112`: active.
- Jetson `100.115.99.8`: active.
- eNodeB direct Tailscale `100.69.186.34`: offline, last seen `53d ago`.
- eNodeB SSH should use EPC as jump host and target `tp2@10.10.10.2`.
- Read-only EPC reachability check after correcting an operator typo from `10.0.0.2` to `10.10.10.2`:
  - `10.0.0.2`: ping failed, SSH port check failed.
  - `10.10.10.2`: ping ok, SSH port check ok.
- The orchestration config now uses `local` for EPC operations and `10.10.10.2` as the eNodeB SSH target when running from the EPC.
- No car Tailscale host is currently documented in `MACHINES.md` or visible in the checked status output.

EPC read-only SSH check:

- Hostname: `tp2-EPC`.
- User: `tp2`.
- `srsepc` binary found at `/usr/local/bin/srsepc`.
- `mosquitto` binary found at `/usr/sbin/mosquitto`.
- `mosquitto_pub` binary found at `/usr/bin/mosquitto_pub`.
- `/home/tp2/TP2_red4G` exists.

Jetson read-only SSH check:

- Hostname: `tp2-jetson`.
- User: `grupo4`.
- `tp2-roboflow-inference.service`: active and enabled.
- `http://127.0.0.1:9001/openapi.json`: reachable locally on Jetson.
- Follow-up from the EPC execution context:
  - Tailscale SSH from EPC to Jetson was denied by tailnet policy.
  - The orchestration now treats Jetson SSH as optional and validates `http://100.115.99.8:9001/openapi.json` from the EPC instead.

Initial eNodeB read-only SSH check:

- Direct Tailscale SSH to `tp2@100.69.186.34` timed out.
- Follow-up SSH through EPC to `tp2@10.10.10.2` succeeded:
  - hostname: `tp2-ENB`
  - user: `tp2`
  - `tp2-srsenb.service`: inactive or not installed yet.
- eNodeB autostart units were not installed or started during this initial read-only check.

Post-install read-only validation from EPC:

- `ops/bin/tp2-validate` completed successfully.
- EPC:
  - `tp2-srsepc.service`: installed, disabled, inactive.
  - `tp2-local-inference.service`: installed, disabled, inactive.
  - `tp2-car-control.service`: installed, disabled, inactive.
  - `tp2-car-command-am-cloud.service`: installed as static oneshot.
  - `mosquitto.service`: enabled and active.
- eNodeB:
  - `tp2-enb-link.service`: installed and enabled, inactive until next eNodeB boot or explicit manual start.
  - `tp2-bladerf-fpga.service`: installed, inactive.
  - `tp2-srsenb.service`: installed, inactive.
  - `tp2-enb-link.service` runs `/home/tp2/to_epc_link.sh` at boot and is no longer started by `tp2-up`.
- Jetson:
  - `http://100.115.99.8:9001/openapi.json` was reachable from EPC.

`ops/bin/tp2-status` read-only output summary:

- EPC:
  - `tp2-srsepc.service`: installed and inactive.
  - `mosquitto.service`: active.
  - `tp2-local-inference.service`: installed and inactive.
  - `tp2-car-control.service`: installed and inactive.
  - MQTT port `1883` is listening.
- eNodeB:
  - reachable through EPC jump path to `tp2@10.10.10.2`.
  - `tp2-enb-link.service`: boot-time service for `/home/tp2/to_epc_link.sh`; enabled and inactive until the next eNodeB boot.
  - `tp2-bladerf-fpga.service`: installed and inactive.
  - `tp2-srsenb.service`: installed and inactive.
- Jetson:
  - direct service status is available only when `TP2_JETSON_SSH` is configured.
  - default EPC-run mode validates the Jetson HTTP endpoint from EPC.
- Car:
  - car-side runtime is operator-managed and intentionally not controlled by `tp2-up`.
  - EPC orchestration waits for UE IP `172.16.0.2` before publishing `AM-Cloud`.

## Remaining Work

- eNodeB is not reachable directly on Tailscale, but it is reachable through the EPC to `10.10.10.2`.
- After installation, `tp2-enb-link.service` is enabled on eNodeB so `/home/tp2/to_epc_link.sh` runs at boot and is no longer started explicitly by `tp2-up`.
- `tp2-up` now validates eNodeB -> EPC backhaul reachability instead of requiring or starting `tp2-enb-link.service`.
- Full runtime validation must be done later with eNodeB online, bladeRF connected, LTE attach active, car UE present, and safe car-side fallback confirmed.
