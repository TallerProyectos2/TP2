# 2026-04-28 - Manual/autonomous mode guard

## Scope

Fix `servicios/coche.py` so manual browser control does not arm or move the car while idle, and so stale/manual browser events cannot switch an active autonomous session back to manual.

## Local validation

- `python -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/roboflow_runtime.py`
- `PYTHONPATH=servicios python -m unittest tests/test_autonomous_driver.py tests/test_coche_runtime.py`
  - result: 14 tests passed
- Local HTTP smoke with `TP2_ENABLE_INFERENCE=0`:
  - neutral `POST /control` in manual returned `armed=false`, `source=neutral`
  - after `POST /mode autonomous`, stale `POST /control` returned `mode=autonomous`
  - `POST /control/neutral` preserved `mode=autonomous`
  - `POST /control/stop` returned `mode=manual`, `source=stop`
- Served UI JavaScript was extracted from `/` and checked with `node --check`.

## Operational note

`/control/neutral` is now manual-control release. It does not leave autonomous mode. The explicit operator stop path is `/control/stop`, used by the web Stop button.
