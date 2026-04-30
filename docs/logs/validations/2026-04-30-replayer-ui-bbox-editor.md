# 2026-04-30 - Replayer UI rework and manual bbox editor

## Scope

- Reworked the offline replayer UI in `servicios/session_replayer.py`:
  - Viewport-fit layout: header + main occupy 100vh, the frame stage is sized
    to the loaded image's natural aspect ratio so the deck and timeline always
    remain on screen without the previous black gap below the video.
  - Compact deck with grouped transport controls, edit-mode toggle, speed
    selector, in-deck timeline, and tabular position readout.
  - Sidebar polish: status badges per session, review-progress bar, in-card
    glyphs, manual/model label icons, Space Grotesk display font for the
    brand, IBM Plex Sans/Mono retained as the body and data fonts.
  - Toast notifications, keyboard hints, and Escape/Delete shortcuts.
- Added a manual bounding-box editor for frames without detections:
  - SVG overlay over the raw frame image when "Cuadros" mode is on.
  - Click-and-drag creates a manual bbox using the currently selected
    relabel class.
  - Existing model bboxes are rendered read-only with a dashed stroke;
    manual bboxes are editable and deletable.
  - Backend stores manual labels under `manual_labels.json` plus a
    `manual_labels.jsonl` audit trail. Manual labels are merged into
    `frame_payload()` and rendered in `draw_overlay()` in cyan.
  - New endpoints `POST /api/frame/box` and `POST /api/frame/box/delete`
    handle create-or-update and delete with the same session-resolution
    safety as existing endpoints.

## Local validation

Commands run from repo root:

```bash
python -m py_compile servicios/session_replayer.py
PYTHONPATH=servicios python -m unittest tests.test_session_replayer -v
```

Result:

- No Python compile errors.
- All 5 existing replayer tests passed.

End-to-end smoke test (ad-hoc Python harness):

- Created a temporary session with a single frame and no labels.
- `POST /api/frame/box` with `bbox_xyxy=[10,20,100,200]` and class `STOP`
  returned the persisted manual label with a generated id and source
  `manual`.
- `GET /api/frame?session=...&idx=0` returned the manual label merged
  into the frame `labels` list with `source: "manual"` and an index
  starting at 10000.
- `POST /api/frame/box/delete` with the returned id removed the manual
  label from the merged listing on the next frame fetch.
- `manual_labels.json` and `manual_labels.jsonl` were both written under
  the session directory.

## Compatibility

- `manifest.jsonl` is not modified by manual labels.
- Existing `labels_reviewed.json` flow is unchanged.
- The HTML still serves at `GET /` and the `/api/sessions`,
  `/api/session`, `/api/frame`, `/frame.jpg`, `/video.mp4` endpoints
  retain their previous shape, with `frame_payload` now also exposing
  `model_label_count` and `manual_label_count` for the UI counters.

## Out of scope

- No firmware changes.
- No EPC live runtime changes.
- No remote-machine actions; this validation is local-only against a
  temporary session directory.
