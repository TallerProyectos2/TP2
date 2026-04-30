# TP2 Automated Startup Orchestration

## Scope

This document defines the repository-owned startup layer for the full TP2 lab.
It keeps the existing architecture boundaries:

- EPC remains the orchestration and control hub.
- eNodeB remains radio-only.
- Jetson remains inference-only.
- Car-side runtime is operator-managed; EPC automation only checks for UE IP before publishing state.

The automation is intentionally based on `systemd` plus SSH orchestration from the EPC.
Docker is used only where it already fits the runtime contract: Jetson Roboflow inference.

## Files

- `ops/tp2-lab.env.example`: non-secret lab defaults.
- `ops/bin/tp2-up`: starts a full session in order.
- `ops/bin/tp2-down`: stops a session in reverse order.
- `ops/bin/tp2-status`: reads current service/process/port state.
- `ops/bin/tp2-validate`: read-only checks for installed units and reachable endpoints.
- `ops/bin/tp2-install-systemd`: installs the versioned systemd units on the target machines.
- `ops/bin/tp2-install-sudoers`: installs narrow passwordless sudo rules for TP2 `systemctl` operations.
- `ops/bin/tp2-enb-load-fpga`: eNodeB helper that waits for `bladeRF` and loads `/home/tp2/Descargas/hostedxA9.rbf`.
- `ops/systemd/`: unit templates for EPC, eNodeB, and Jetson.
- `ops/sudoers/`: sudoers templates with no passwords or tokens.

## Configuration

Runtime configuration should live outside the repository:

- `/etc/tp2/lab.env` for machine-wide orchestration settings.
- `~/.config/tp2/lab.env` for operator-local overrides.
- `/home/tp2/.config/tp2/inference.env` or `/home/tp2/.config/tp2/coche-jetson.env` for EPC Roboflow runtime variables.
- `/etc/tp2/jetson-inference.env` for Jetson Roboflow container secrets.

Do not put SSH passwords, Roboflow API keys, or tokens in repository files.
The startup scripts use passwordless sudo only for the specific TP2 `systemctl` commands listed in `ops/sudoers/`.

Run the orchestration commands from the EPC repository checkout:

- EPC: `local`
- eNodeB: `tp2@10.10.10.2`
- Jetson HTTP endpoint: `http://100.115.99.8:9001`
- Jetson SSH: optional. Leave `TP2_JETSON_SSH` empty when the EPC can reach the HTTP endpoint but Tailscale policy does not allow SSH.
- Default startup profile: `jetson`, so `ops/bin/tp2-up` and `ops/bin/tp2-up --profile jetson` are equivalent unless `TP2_DEFAULT_PROFILE` is overridden.
- Live video web view: `http://100.97.19.112:8088/` from Tailscale, or `http://172.16.0.1:8088/` from the UE side when reachable.
- Car UE gating: disabled by default. `tp2-up` checks the car UE best-effort and continues if not confirmed. Set `TP2_REQUIRE_CAR_UE=1` to restore the old blocking wait.

If you intentionally run from an operator laptop instead, create a local override file that sets:

```bash
TP2_EPC_SSH=tp2@100.97.19.112
TP2_ENB_SSH=tp2@10.10.10.2
TP2_ENB_SSH_PROXY=tp2@100.97.19.112
```

The car runtime is intentionally outside this automation. Operators start and stop the car-side process manually. `tp2-up` does not restart the car-side service unless `TP2_RESTART_CAR_ON_UP=1` is explicitly set in host-local config.
The EPC automation no longer blocks on UE IP detection by default because the modem attach state can be valid even when the previous fixed target `172.16.0.2` does not answer ping or the attach log is not fresh. The live HSS was observed with dynamic UE allocation on `2026-04-27`, so confirm the current IP from the latest `srsepc` log when troubleshooting.

## Startup Order

`ops/bin/tp2-up` performs this order. The default profile is `jetson`.

1. Verify access to EPC and eNodeB. Jetson SSH is checked only when `TP2_JETSON_SSH` is set and `TP2_START_JETSON_INFERENCE=1`.
2. Check eNodeB backhaul reachability to EPC. `tp2-enb-link.service` is enabled at eNodeB boot and wraps `/home/tp2/to_epc_link.sh`, but `tp2-up` does not start it.
3. On eNodeB, run `tp2-bladerf-fpga.service`, which waits for `bladeRF` and loads `/home/tp2/Descargas/hostedxA9.rbf`.
4. On EPC, start `tp2-srsepc.service`.
5. Wait for EPC S1/GTP sockets and `srsepc`.
6. On eNodeB, start `tp2-srsenb.service`.
7. Check the car UE path once: ping the configured `TP2_CAR_UE_IP` or inspect the latest EPC attach log for IMSI `901650000052126`. Continue when not confirmed unless `TP2_REQUIRE_CAR_UE=1`.
8. Check the Jetson inference endpoint from EPC. If `TP2_JETSON_SSH` and `TP2_START_JETSON_INFERENCE=1` are configured, start `tp2-roboflow-inference.service` first.
9. On EPC, start `tp2-local-inference.service` when local fallback is enabled.
10. On EPC, start Mosquitto with `sudo systemctl start mosquitto`.
11. On EPC, start `tp2-car-control.service`.
12. Re-check the car UE IP once before publishing `AM-Cloud`. Continue when not confirmed unless `TP2_REQUIRE_CAR_UE=1`. Restart the car-side systemd service only when `TP2_RESTART_CAR_ON_UP=1`.
13. Check the EPC live video web endpoint.
14. On EPC, ensure the retained car mode with `ops/bin/tp2-mqtt-ensure-car-mode`. It publishes retained `AM-Cloud` only when the retained state is missing or different, then verifies the retained value. The legacy `tp2-car-command-am-cloud.service` now runs the same idempotent helper.

Profiles:

- `jetson`: use Jetson inference and keep EPC local inference available as fallback.
- `local`: start the EPC local inference path and set the systemd manager environment for `tp2-car-control.service` to `127.0.0.1:9001`.
- `no-inference`: bring up LTE/control path without explicitly starting inference services. Use only with a car-control environment that disables live inference.

If a host-local `EnvironmentFile` sets the same inference variables, inspect the resulting `systemctl show tp2-car-control.service -p Environment` output before relying on a profile switch.

## Shutdown Order

`ops/bin/tp2-down` performs the reverse operational sequence:

1. Stop EPC car control.
2. Stop EPC local inference fallback.
3. Optionally stop Mosquitto if `TP2_STOP_MOSQUITTO_ON_DOWN=1`.
4. Stop eNodeB `srsenb`.
5. Stop EPC `srsepc`.
6. Optionally stop Jetson inference if `TP2_STOP_JETSON_ON_DOWN=1`.

The default keeps Mosquitto and Jetson inference alive because they are safe support services and may be shared across checks.

Mosquitto start/stop operations must use the broker unit directly (`sudo systemctl start mosquitto` and `sudo systemctl stop mosquitto`). The repository sudoers template allows both `mosquitto` and `mosquitto.service` spellings for compatibility, but `tp2-up`/`tp2-down` default to `mosquitto`.

## Installation

The installer copies unit templates and helper scripts, then runs `systemctl daemon-reload`.
It does not start the radio/control lab services.

```bash
cd /home/tp2/TP2_red4G
ops/bin/tp2-install-sudoers all
ops/bin/tp2-install-systemd epc
ops/bin/tp2-install-systemd enb
ops/bin/tp2-install-systemd jetson
```

Use `ops/bin/tp2-install-systemd all` only when EPC, eNodeB, and Jetson SSH access are all healthy.
The installer enables `tp2-enb-link.service` on eNodeB so `/home/tp2/to_epc_link.sh` runs at every eNodeB boot. It does not start the radio services.
`ops/bin/tp2-install-sudoers all` may prompt for the sudo password once per machine; it installs only narrow command permissions and does not store the password.

If the eNodeB account does not have passwordless sudo, install the eNodeB units once from the EPC with an interactive SSH session:

```bash
scp ops/bin/tp2-enb-load-fpga tp2@10.10.10.2:/tmp/
scp ops/systemd/enb/*.service tp2@10.10.10.2:/tmp/
ssh -t tp2@10.10.10.2 'sudo install -m 0755 /tmp/tp2-enb-load-fpga /usr/local/sbin/tp2-enb-load-fpga && sudo install -m 0644 /tmp/tp2-enb-link.service /etc/systemd/system/tp2-enb-link.service && sudo install -m 0644 /tmp/tp2-bladerf-fpga.service /etc/systemd/system/tp2-bladerf-fpga.service && sudo install -m 0644 /tmp/tp2-srsenb.service /etc/systemd/system/tp2-srsenb.service && sudo systemctl daemon-reload && sudo systemctl enable tp2-enb-link.service'
```

That command enables only the boot-time link service; it does not start `srsenb`.

After installing, inspect before starting:

```bash
ops/bin/tp2-validate
ops/bin/tp2-status
```

## Live Web View

`tp2-car-control.service` runs `coche.py` with the web view and remote manual control enabled by default:

- bind: `0.0.0.0:8088`
- status endpoint: `http://127.0.0.1:8088/status.json` on EPC
- operator URL over Tailscale: `http://100.97.19.112:8088/`
- camera stream: `/video.mjpg`
- control API: `POST /control` applies non-neutral browser commands only while the runtime is in `manual`; neutral manual posts stay unarmed; `POST /control/neutral` releases manual control without leaving `autonomous`; `POST /control/stop` forces manual neutral stop
- steering trim API: `POST /steering-trim` changes the live compensation value used before UDP send
- cruise API: `POST /cruise-speed` changes the live autonomous forward throttle
- turn compensation API: `POST /turn-compensation` enables/tunes the optional periodic right steering pulse
- mode API: `POST /mode` switches between `manual` and `autonomous`; manual remains the safe startup mode
- recording API: `POST /recording` starts/stops dataset capture; `GET /recording.json` reports output path and counters
- safety timeout: `TP2_WEB_CONTROL_TIMEOUT_SEC` forces neutral when browser commands stop
- command stream: `TP2_CONTROL_TX_HZ` sends the latest steering/throttle to the last car UDP endpoint while it is fresh

OpenCV desktop windows are not part of the unattended runtime; the browser view is the operator surface.

## Safety Notes

- Loading the FPGA bitstream with `bladeRF-cli -l` is not a firmware update.
- The automation must not run `bladeRF` firmware updates.
- eNodeB units host only radio/backhaul work.
- Jetson units host only inference work.
- The car must retain safe fallback behavior when commands are missing or delayed.
- If eNodeB is offline on Tailscale, do not claim the full startup path validated.
