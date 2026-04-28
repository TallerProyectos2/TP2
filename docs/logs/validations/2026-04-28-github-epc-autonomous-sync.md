# 2026-04-28 - GitHub and EPC Autonomous Sync

## Scope

Resolve the divergent `main` state after the EPC sync was pushed without the autonomous driver work.

Integrated sources:

- `origin/main` at `c60e264` with the EPC sync.
- `origin/backup/pre-epc-sync-20260428` at `057394d` with the pre-sync backup line.
- Local autonomous driver changes for `servicios/coche.py`, `servicios/autonomous_driver.py`, tests, and docs.

## GitHub Result

- Local `main` merged `origin/main` and the backup branch.
- Add/add conflict in `scripts_profesor/car1_grupo4.py` was resolved with identical content and executable mode `100755`.
- Pushed `main` to GitHub:
  - before: `c60e264`
  - after: `b2bce61`

## Local Validation

```bash
python3 -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/roboflow_runtime.py
PYTHONPATH=servicios python3 -m unittest tests/test_autonomous_driver.py
bash -n ops/bin/tp2-status ops/lib/tp2-common.sh ops/bin/tp2-up ops/bin/tp2-validate
```

Result: OK.

Local runtime smoke test with isolated ports and inference disabled:

- `POST /control` returned `mode=manual`.
- `POST /mode` accepted `autonomous`.
- `GET /status.json` returned `control.mode=autonomous`, `video.has_video=true`, `udp.packets.I=2`, and `autonomy.decision.action=safe-neutral`.

## EPC Result

EPC checkout:

```bash
cd /home/tp2/TP2_red4G
git pull --ff-only
```

Result:

- EPC fast-forwarded from `c60e264` to `b2bce61`.
- `git status --short --branch` returned clean `main...origin/main`.
- `servicios/autonomous_driver.py` is present on EPC.
- `scripts_profesor/car1_grupo4.py` remains executable (`100755`).

Remote EPC validation:

```bash
python3 -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/roboflow_runtime.py
PYTHONPATH=servicios python3 -m unittest tests/test_autonomous_driver.py
bash -n ops/bin/tp2-status ops/lib/tp2-common.sh ops/bin/tp2-up ops/bin/tp2-validate
```

Result: OK on EPC Python `3.10.12`.

Runtime note:

- `tp2-car-control.service` was `inactive`.
- No `coche.py` listener was present on `20001/UDP` or `8088/TCP`.
- No live runtime restart was performed.
