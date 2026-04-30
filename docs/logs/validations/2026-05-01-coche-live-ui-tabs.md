# 2026-05-01 - Live coche.py UI rework with tabbed tools panel

## Scope

- Refactored `LIVE_VIEW_HTML` in `servicios/coche.py` to align with the
  retraining replayer aesthetic and to make the tools panel more useful.
- Added a sticky 2x2 context strip at the top of the right panel showing
  Latencia IA, FPS Video, Detecciones, and Accion (with action color
  tagging: continue/stop/turn/yield).
- Reorganized the right column into 4 tabs:
  - `Telemetria`: Inferencia card (sparklines, detections list) + Autonomia
    card (decision rows). Default tab.
  - `Tuning`: Marcha (cruise) + Pulso derecha + Compensacion (steering
    trim) with consistent slider styling and a footer block for
    `Guardar como default`.
  - `Dataset`: Dataset metrics + a callout pointing the operator to the
    Grabar / Revisar buttons that live in the deck.
  - `Sistema`: Enlace 4G + Sistema diagnostics (origen control, watchdog,
    stream clients, posts control).
- Tab badges update live (detection count on Telemetria, current cruise
  setpoint on Tuning, REC/OFF on Dataset, error count on Sistema).
- Polished the deck:
  - New mode toggle with icons for Manual / Autonomo.
  - Compact action buttons (`btn-record`, `btn-review`, `stop`) right
    aligned, with the Stop button using the display font and a hot red
    treatment.
  - Sliders share the same gradient progress fill and ring thumb as the
    replayer timeline.
- Extra keyboard shortcuts: `M` for manual mode, `N` for autonomous mode,
  `Tab` / `Shift+Tab` to cycle right-panel tabs, `Escape` to trigger the
  emergency stop.
- Display font: Space Grotesk added for the brand and tab labels; IBM
  Plex Sans/Mono retained for body and tabular data.

## Bugs caught and fixed

- The new record button was originally given `class="action rec"`. The
  `.rec` selector was already used by the absolute-positioned EN VIVO
  badge inside the video, so the record button was being lifted into the
  top-right of the page. Renamed the new classes to `btn-record` and
  `btn-review` to avoid the cascade collision.
- All existing JS DOM ids and the `pollStatus` mapping were kept intact,
  so no backend changes were required.

## Local validation

```bash
python -m py_compile servicios/coche.py
PYTHONPATH=servicios python -m unittest discover -s tests
node -e "..."  # parsed inline JS through `new Function(scriptBody)`
```

Result:

- Compile clean, all 52 tests pass.
- Inline JS parsed without syntax errors via Node.
- Live smoke test: `coche.py` started locally with
  `TP2_INFERENCE_BACKEND=disabled`, `TP2_LANE_ASSIST_ENABLE=0`,
  `TP2_SESSION_RECORD_AUTOSTART=0`. Visited `GET /` in Chrome and
  confirmed:
  - Header pills, both clocks (Sesion, Hora) render correctly.
  - Stage video shell shows the "Sin senal" pulse with the UDP bind hint.
  - HUD chips (FPS, Lat, Det, Frame) overlayed on the video.
  - Deck instruments (steer meter, WASD keys, throttle meter) align.
  - Deck actions row shows Manual/Autonomo, Grabar, Revisar, Stop in the
    correct positions.
  - Right panel context strip shows Latencia IA/FPS Video/Detecciones/Accion
    with the cyan action dot.
  - Tabs `Telemetria` / `Tuning` / `Dataset` / `Sistema` switch panels
    with animation.
  - Tuning sliders for Marcha and Compensacion have the gradient progress
    fill aligned to the live value, and the Pulso derecha toggle and
    compact fields render correctly (no more checkbox stretching).
  - Dataset tab shows live dataset stats with the right typography.
  - Sistema tab shows ENLACE 4G + SISTEMA cards.

## Compatibility

- All HTTP routes in `coche.py` are unchanged.
- All JSON shapes consumed by the UI are unchanged.
- All DOM ids referenced by the polling loop are kept, so the existing
  status fan-out continues to work.
- Existing keyboard shortcuts (WASD, Space, X, arrows) are kept; new
  shortcuts are additive.

## Out of scope

- No firmware changes.
- No EPC service restart performed in this validation; only the local
  laptop ran `coche.py` against a private UDP bind.
- No remote machine actions.
