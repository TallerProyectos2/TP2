# Temporary Speed Sign Policy Local Validation

- Date: 2026-05-15
- Scope: `servicios/autonomous_driver.py`, `servicios/coche.py`
- Context: local validation before EPC deployment.

## Change Validated

- `VELOCIDAD-MAX-30` applies throttle `0.50` for `3.0 s`.
- `VELOCIDAD-MAX-90` applies throttle `0.70` for `3.0 s`.
- After the temporary speed window, the controller returns to the live cruise throttle selected by the web slider.
- The same visible speed-sign track does not refresh the timer indefinitely.
- The `Vel s` runtime setting exposes `TP2_AUTONOMOUS_SPEED_OVERRIDE_SEC` in the web tuning UI.

## Local Checks

```text
python -m py_compile servicios/autonomous_driver.py servicios/coche.py servicios/session_replayer.py
OK
```

```text
PYTHONPATH=servicios python -m unittest discover -s tests -p 'test_autonomous_driver.py'
Ran 18 tests
OK
```

```text
PYTHONPATH=servicios python -m unittest discover -s tests -p 'test_coche_runtime.py'
Ran 26 tests
OK
```

```text
PYTHONPATH=servicios python -m unittest discover -s tests
Ran 69 tests
OK
```

```text
git diff --check
OK
```
