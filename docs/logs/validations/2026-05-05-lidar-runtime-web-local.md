# LiDAR Runtime And Web UI Local Validation

## Scope

- Local repo validation for LiDAR support in `servicios/coche.py`.
- No remote EPC, eNodeB, Jetson, or car service was restarted.
- No firmware was touched.

## Change

- Added `servicios/lidar_processor.py` for LiDAR payload normalization, frontal obstacle safety, and reduced point-cloud status output.
- Extended `servicios/coche.py` to accept UDP packet type `L` and LiDAR nested in telemetry `D`.
- Added autonomous LiDAR safety actions:
  - `lidar-stop` for frontal obstacles inside `TP2_LIDAR_STOP_DISTANCE_M`.
  - `lidar-slow` for frontal obstacles inside `TP2_LIDAR_SLOW_DISTANCE_M`.
  - `lidar-caution` for the caution band.
- Added a web stage toggle between camera and LiDAR reconstruction.
- Updated operational docs for the new UDP and UI contract.

## Local Validation

Commands run from repo root:

```text
python -m py_compile servicios/lidar_processor.py servicios/coche.py
PYTHONPATH=servicios python -m unittest discover -s tests -p 'test_lidar_processor.py'
PYTHONPATH=servicios python -m unittest discover -s tests -p 'test_coche_runtime.py'
PYTHONPATH=servicios python -m unittest discover -s tests
node - <<'NODE'
const fs = require('fs');
const src = fs.readFileSync('servicios/coche.py', 'utf8');
const match = src.match(/<script>([\s\S]*?)<\/script>/);
if (!match) throw new Error('script block not found');
new Function(match[1]);
console.log('live ui script parses');
NODE
git diff --check
TP2_BIND_IP=127.0.0.1 TP2_BIND_PORT=23001 TP2_WEB_HOST=127.0.0.1 TP2_WEB_PORT=18088 TP2_ENABLE_INFERENCE=0 TP2_SESSION_RECORD_AUTOSTART=0 python servicios/coche.py
curl -fsS http://127.0.0.1:18088/healthz
curl -fsS http://127.0.0.1:18088/status.json
```

Results:

```text
test_lidar_processor.py: Ran 5 tests, OK
test_coche_runtime.py: Ran 22 tests, OK
full unittest discovery: Ran 59 tests, OK
live UI JavaScript parse check: OK
git diff whitespace check: OK
local HTTP smoke: `/healthz` returned `{"ok":true}` and `/status.json` included `lidar.status=searching` in manual mode
```

`pytest` was attempted first, but the active local Python environment did not have `pytest` installed:

```text
No module named pytest
```

## Remaining Real-Lab Validation

- Connect the physical LiDAR on the car and confirm the actual emitted payload format.
- Confirm EPC receives `L` or LiDAR-bearing `D` packets over LTE.
- Confirm `/status.json` reports live `lidar.frames`, `lidar.status`, `lidar.safety.min_front_distance_m`, and capped `lidar.points`.
- Open `http://100.97.19.112:8088/`, switch to LiDAR view, and visually confirm reconstruction.
- In autonomous mode, validate `lidar-stop` and `lidar-slow` with a controlled obstacle before any free-driving test.
