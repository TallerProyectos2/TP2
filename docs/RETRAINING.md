# TP2 Retraining And Session Recording

## Purpose

The EPC runtime records each normal car session so the team can expand the Roboflow dataset with real LTE driving data and isolate critical situations for manual review.

The recorder is diagnostic. It must never block UDP control or autonomous safety fallback.

## Runtime Capture

Normal EPC sessions run `servicios/coche.py` through `tp2-car-control.service`.

The service template enables:

- `TP2_SESSION_RECORD_AUTOSTART=1`
- `TP2_SESSION_RECORD_VIDEO=1`
- `TP2_SESSION_RECORD_CRITICAL_IMAGES=1`

Default output root:

```bash
/srv/tp2/frames/autonomous
```

Each session creates a timestamped directory with:

- `session.json`: capture configuration and critical-rule metadata.
- `manifest.jsonl`: frame-level source of truth for replay, labels, predictions, autonomy, control, and video index.
- `labels.jsonl`: model-estimated label candidates. These are not ground truth.
- `critical.jsonl`: subset index for frames requiring review.
- `images/frame_*.jpg`: raw saved frames when image capture is enabled.
- `critical/frame_*.jpg`: annotated critical frames.
- `session.mp4`: annotated video with predictions, track ids, autonomous action, and critical-rule badges.
- `labels_reviewed.json` and `labels_reviewed.jsonl`: created by the offline replayer after human review.

Secrets are not written to session directories.

## Roboflow Inference Path

Live inference uses the Roboflow `inference_sdk` HTTP client directly with OpenCV NumPy frames.

The live path no longer writes a temporary JPEG before calling inference. The SDK accepts NumPy arrays for `InferenceHTTPClient.infer(...)`, which avoids filesystem latency and SSD wear during live sessions.

`servicios/inferencia.py` still supports file-path inference for the known-image validation flow.

## Critical Frame Rules

`servicios/coche.py` flags frames for review when any rule triggers:

- confidence in `[0.35, 0.55]`
- same recorder `track_id` changes class on consecutive frames
- a detection appears and disappears in fewer than `TP2_SESSION_RECORD_DISAPPEAR_FRAMES` frames, default `3`
- autonomous decision is ambiguous
- operator overrides or attempts manual control during autonomous mode

Recorder track ids are for dataset triage only. They are not persisted as car-control state.

## Offline Replayer

The normal operator path is from the live `coche.py` web UI:

1. Open `http://100.97.19.112:8088/`.
2. Press `Revisar dataset`.
3. The live runtime starts the replayer on `TP2_SESSION_REPLAYER_PORT`, default `8090`.
4. The replayer opens with a session selector populated directly from `TP2_SESSION_RECORD_DIR`.

The standalone path is still available from any machine with the TP2 Python environment and access to the recording root:

```bash
cd /home/tp2/TP2_red4G
conda activate tp2
python servicios/session_replayer.py /srv/tp2/frames/autonomous --host 0.0.0.0 --port 8090
```

From an operator laptop over Tailscale, open:

```text
http://100.97.19.112:8090/
```

The replayer supports:

- selecting any session directory under the recording root
- stepping through all frames or critical frames only
- viewing overlays from candidate labels and critical rules
- relabeling a detection class
- marking detections valid or rejected
- writing reviewed labels without modifying the original manifest

Reviewed labels are saved in:

```text
labels_reviewed.json
labels_reviewed.jsonl
```

## Retraining Flow

1. Start the normal EPC runtime and run the car session.
2. Confirm a session directory was created under `/srv/tp2/frames/autonomous`.
3. Review `critical.jsonl` first with `session_replayer.py`.
4. Review non-critical frames if more coverage is needed.
5. Treat `labels.jsonl` as model candidates only.
6. Use `labels_reviewed.json` as the curated source for Roboflow upload/export.
7. Train a new Roboflow version.
8. Update only host-local runtime configuration to point `ROBOFLOW_MODEL_ID` at the new model version.
9. Validate with known-image inference, then live EPC status, then a controlled car session.

Do not commit API keys or host-local Roboflow environment files.

## Minimum Validation

For a recorder change:

- `python -m py_compile servicios/coche.py servicios/roboflow_runtime.py servicios/session_replayer.py`
- `PYTHONPATH=servicios python -m unittest discover -s tests`
- Start `coche.py` locally with inference disabled and a temporary record directory.
- Send a synthetic `I + pickle(jpeg)` frame.
- Confirm `manifest.jsonl`, `labels.jsonl`, and `session.mp4` are created.
- Start the replayer from `coche.py` with `POST /replayer/start`.
- Confirm `GET /api/sessions` shows the recorded session and `GET /api/frame?idx=0` returns frame metadata.
