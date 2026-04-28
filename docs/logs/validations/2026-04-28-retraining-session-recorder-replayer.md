# 2026-04-28 - Retraining Session Recorder And Replayer

## Scope

Implemented the EPC-owned retraining/session capture path for `servicios/coche.py`:

- live Roboflow inference uses `inference_sdk` with OpenCV NumPy frames instead of writing a temporary JPEG
- session capture writes `manifest.jsonl`, `labels.jsonl`, `critical.jsonl`, raw frames, critical images, and annotated `session.mp4`
- critical flags cover low confidence, class changes on the same recorder track, short-lived detections, ambiguous autonomous decisions, and operator overrides during autonomous mode
- `servicios/session_replayer.py` provides offline frame replay and relabeling into `labels_reviewed.json`

## Files Changed

- `servicios/roboflow_runtime.py`
- `servicios/coche.py`
- `servicios/session_replayer.py`
- `tests/test_coche_runtime.py`
- `tests/test_roboflow_runtime.py`
- `ops/systemd/epc/tp2-car-control.service`
- `ops/tp2-lab.env.example`
- `PLAN.md`
- `ARCHITECTURE.md`
- `RUNBOOK.md`
- `docs/EPC.md`
- `docs/INFERENCE.md`
- `docs/CAR-AGENT.md`
- `docs/RETRAINING.md`

## Local Validation

Syntax:

```bash
PYTHONPATH=servicios /Users/mario/miniconda3/envs/test/bin/python -m py_compile \
  servicios/coche.py servicios/roboflow_runtime.py servicios/session_replayer.py \
  tests/test_coche_runtime.py tests/test_roboflow_runtime.py
```

Result: OK.

Unit tests:

```bash
PYTHONPATH=servicios /Users/mario/miniconda3/envs/test/bin/python -m unittest discover -s tests
```

Result:

```text
Ran 22 tests in 0.003s
OK
```

Runtime smoke:

```bash
PYTHONPATH=servicios \
TP2_BIND_IP=127.0.0.1 \
TP2_BIND_PORT=29001 \
TP2_WEB_HOST=127.0.0.1 \
TP2_WEB_PORT=18088 \
TP2_ENABLE_INFERENCE=0 \
TP2_SESSION_RECORD_AUTOSTART=1 \
TP2_SESSION_RECORD_DIR=/tmp/tp2-record-smoke \
TP2_SESSION_RECORD_MIN_INTERVAL_SEC=0 \
TP2_SESSION_RECORD_VIDEO=1 \
/Users/mario/miniconda3/envs/test/bin/python -u servicios/coche.py
```

Synthetic UDP frame:

- sent `I + pickle(jpeg)` to `127.0.0.1:29001`
- received control packet beginning with `C`, length `17`
- `GET /status.json` reported `video.has_video=true`, `video.frames=1`, `inference.enabled=false`

Recorder outputs:

```text
/tmp/tp2-record-smoke/<session>/README.txt
/tmp/tp2-record-smoke/<session>/images/frame_00000001.jpg
/tmp/tp2-record-smoke/<session>/labels.jsonl
/tmp/tp2-record-smoke/<session>/manifest.jsonl
/tmp/tp2-record-smoke/<session>/session.json
/tmp/tp2-record-smoke/<session>/session.mp4
```

Offline replayer:

```bash
PYTHONPATH=servicios /Users/mario/miniconda3/envs/test/bin/python -u \
  servicios/session_replayer.py /tmp/tp2-record-smoke/<session> \
  --host 127.0.0.1 --port 18089
```

Checks:

- `GET /api/session` returned `frames=1`
- `GET /api/frame?idx=0` returned the manifest frame
- `GET /frame.jpg?idx=0` returned JPEG bytes beginning with `ffd8ffe0`

## EPC Read-Only State

Before deployment checks:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 tp2@100.97.19.112 \
  'hostname; cd /home/tp2/TP2_red4G && git status --short --branch; \
   systemctl is-active tp2-car-control.service 2>/dev/null || true'
```

Observed:

```text
tp2-EPC
## main...origin/main
inactive
```

`http://127.0.0.1:8088/status.json` was not reachable because `tp2-car-control.service` was inactive.

Jetson endpoint `http://100.115.99.8:9001/info` timed out from EPC during this validation window, so live Jetson inference was not claimed as validated for this change.

## Notes

- No firmware changes were made.
- No Roboflow secrets or SSH passwords were written to the repository.
- The EPC service was not restarted during read-only inspection because it was inactive.
