# 2026-04-14 - EPC default Jetson profile, live web view, and sudoers

## Scope

- Made `ops/bin/tp2-up` default to the `jetson` profile through `TP2_DEFAULT_PROFILE=jetson`.
- Added a live web view to `servicios/coche.py` so the EPC control runtime exposes:
  - `/` operator page
  - `/video.mjpg` annotated camera stream
  - `/status.json` control/inference status
- Added narrow `sudoers` templates for passwordless TP2 `systemctl` operations.
- Did not store any password, token, or API key in repository files.
- Did not start LTE radio, `srsepc`, `srsenb`, or the real car control runtime during validation.
- No firmware updates were performed.

## Files Changed

- `ops/tp2-lab.env.example`
- `ops/lib/tp2-common.sh`
- `ops/bin/tp2-up`
- `ops/bin/tp2-status`
- `ops/bin/tp2-validate`
- `ops/bin/tp2-install-sudoers`
- `ops/systemd/epc/tp2-car-control.service`
- `ops/sudoers/epc/tp2-lab`
- `ops/sudoers/enb/tp2-lab`
- `servicios/coche.py`
- `docs/AUTOSTART.md`
- `RUNBOOK.md`
- `MACHINES.md`
- `ARCHITECTURE.md`
- `PLAN.md`
- `docs/EPC.md`
- `docs/INFERENCE.md`
- `docs/CAR-AGENT.md`
- `docs/NETWORK.md`

## Validation

Local syntax checks:

```bash
python3 -m py_compile servicios/coche.py

bash -n ops/lib/tp2-common.sh \
  ops/bin/tp2-enb-load-fpga \
  ops/bin/tp2-up \
  ops/bin/tp2-down \
  ops/bin/tp2-status \
  ops/bin/tp2-validate \
  ops/bin/tp2-install-systemd \
  ops/bin/tp2-install-sudoers
```

Result:

- Python compile check passed.
- Shell syntax check passed.

Remote install/validation from EPC:

- Installed `/etc/sudoers.d/tp2-lab` on EPC after `visudo -cf` parse checks.
- Installed `/etc/sudoers.d/tp2-lab` on eNodeB after `visudo -cf` parse checks.
- Reinstalled `tp2-car-control.service` on EPC and ran `systemctl daemon-reload`.
- `ops/bin/tp2-validate` completed successfully:
  - EPC config and units: ok
  - EPC passwordless TP2 sudo: ok
  - eNodeB config and units: ok
  - eNodeB passwordless TP2 sudo: ok
  - Jetson inference endpoint reachable from EPC: ok
  - LTE socket check: ok or EPC core inactive
- After invalidating sudo timestamps with `sudo -k`, command permission checks returned `0` for:
  - EPC: `/usr/bin/systemctl start tp2-car-control.service`
  - eNodeB: `/usr/bin/systemctl start tp2-bladerf-fpga.service`

Isolated live web test on EPC:

```bash
TP2_BIND_IP=127.0.0.1 \
TP2_BIND_PORT=29001 \
TP2_WEB_PORT=18088 \
TP2_ENABLE_INFERENCE=0 \
TP2_ENABLE_WEB_VIEW=1 \
TP2_ENABLE_OPENCV_WINDOWS=0 \
/home/tp2/miniforge3/bin/conda run --no-capture-output -n tp2 \
  python -u servicios/coche.py
```

Result:

- `http://127.0.0.1:18088/status.json` returned JSON successfully.
- `http://100.97.19.112:18088/status.json` returned JSON successfully from the operator machine over Tailscale.
- Test used loopback UDP port `29001`, so it did not bind the live car control port.
- The isolated test process was stopped and ports `18088/TCP` and `29001/UDP` were confirmed clear afterward.

Current live status after validation:

- EPC:
  - `tp2-srsepc.service`: inactive
  - `mosquitto.service`: active
  - `tp2-local-inference.service`: inactive
  - `tp2-car-control.service`: inactive
  - `8088/TCP`: not listening because real car control was not started
- eNodeB:
  - `tp2-enb-link.service`: enabled and inactive until next eNodeB boot or explicit start
  - `tp2-bladerf-fpga.service`: inactive
  - `tp2-srsenb.service`: inactive
- Jetson endpoint from EPC:
  - `http://100.115.99.8:9001/openapi.json`: ok

## Operator Notes

- Normal startup from EPC is now:
  - `cd /home/tp2/TP2_red4G`
  - `ops/bin/tp2-up`
- The explicit form remains valid:
  - `ops/bin/tp2-up --profile jetson`
- During a live session, open:
  - `http://100.97.19.112:8088/`
