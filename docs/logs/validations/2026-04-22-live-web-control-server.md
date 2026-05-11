# 2026-04-22 - Live Web Control Server

## Objective

Implement and locally validate the car runtime web server to:

- show real-time MJPEG video;
- expose snapshots and JSON status;
- show inference and overlays on the received frame;
- send manual remote control from a browser;
- apply a control watchdog that returns to neutral if the browser stops publishing.

## Locally Validated Changes

- `servicios/coche.py` starts the HTTP server inside the runtime process.
- Available endpoints:
  - `GET /`
  - `GET /status.json`
  - `GET /video.mjpg`
  - `GET /snapshot.jpg`
  - `POST /control`
  - `POST /control/neutral`
- Web control is bounded by:
  - `TP2_ENABLE_WEB_CONTROL`
  - `TP2_WEB_CONTROL_TIMEOUT_SEC`
  - `TP2_WEB_CONTROL_MAX_FORWARD`
  - `TP2_WEB_CONTROL_MAX_REVERSE`
- Web status includes packet counters, last client, last packet type, video frames, inference status, and active control source.

## Local Validation

Compile command:

```console
/Users/mario/miniconda3/envs/test/bin/python -m py_compile servicios/coche.py servicios/roboflow_runtime.py
```

Result: OK, no error output.

Local test server:

```console
TP2_BIND_IP=127.0.0.1 TP2_BIND_PORT=29001 TP2_WEB_HOST=127.0.0.1 TP2_WEB_PORT=18088 TP2_ENABLE_INFERENCE=0 TP2_ENABLE_OPENCV_WINDOWS=0 /Users/mario/miniconda3/envs/test/bin/python -u servicios/coche.py
```

Relevant output:

```console
Live web view listening on http://127.0.0.1:18088/
Manual control server listening on 127.0.0.1:29001
Inference: disabled (local/model) endpoint=http://100.115.99.8:9001
```

HTTP/UDP test with a synthetic JPEG frame sent as UDP `I`:

```json
{
  "control_source": "web",
  "control_status": 200,
  "has_video": true,
  "neutral_source": "web-timeout",
  "neutral_status": 200,
  "packet_total": 1,
  "snapshot": {
    "content_type": "image/jpeg",
    "jpeg_soi": true,
    "status": 200
  },
  "timeout_source": "web-timeout",
  "udp_reply": {
    "bytes": 17,
    "kind": "C",
    "steering": 1.0,
    "throttle": 0.2
  },
  "video_frames": 1
}
```

Corrupt-frame test: the server recorded `bad_image_frames=1`, kept `last_error` in diagnostics, and still responded to the car with a 17-byte `C` packet.

Snapshot test:

```json
{
  "status": 200,
  "content_type": "image/jpeg",
  "jpeg_soi": true,
  "first_hex": "ffd8ffe0"
}
```

Result: the local server accepts web control, responds to the car over UDP with `C`, publishes video/snapshots, and returns to `web-timeout` after the timeout.

## Observed Remote State

Command:

```console
ops/bin/tp2-status
```

Summary:

- EPC: `srsepc` active, `mosquitto` active, `tp2-car-control.service` active.
- EPC: UDP `172.16.0.1:20001` listening.
- EPC: web `0.0.0.0:8088` listening and web endpoint responding.
- Car UE: not confirmed by `tp2-status`.
- eNodeB: link, FPGA, and `srsenb` active.
- Jetson: inference service active and OpenAPI reachable.

Remote query:

```console
curl -fsS --max-time 4 http://100.97.19.112:8088/status.json
```

Observed result: the previous remote deployment responds, but without video frames (`has_video=false`, `video_frames=0`) and with inference waiting.

## Operational Notes

`tp2-car-control.service` was not restarted on EPC during this validation to avoid interrupting the active runtime.

The remote checkout at `/home/tp2/TP2_red4G` was clean, but diverged from `origin/main` (`ahead 1, behind 4`) while the local repo also had pending changes. Activating this version in the real lab requires resolving/synchronizing that divergence and restarting only `tp2-car-control.service` in a controlled window.

## Later Remote Intervention

After observing from the operator interface that EPC was still serving the old version, the new versions of these files were copied:

- `servicios/coche.py`
- `ops/systemd/epc/tp2-car-control.service`

to the remote checkout at `/home/tp2/TP2_red4G`.

The systemd install/restart could not complete because `sudo` requested an interactive password. The old `tp2-car-control.service` process was terminated; systemd left the unit `inactive` instead of relaunching it. To recover the runtime, the new `coche.py` was started manually as user `tp2`:

```console
cd /home/tp2/TP2_red4G/servicios
nohup /home/tp2/miniforge3/bin/conda run --no-capture-output -n tp2 python -u coche.py > /tmp/tp2-car-control-web.log 2>&1 &
```

Remote validation after manual startup:

- `8088/TCP`: active.
- `20001/UDP`: active.
- `GET /` serves the new `TP2 Live Control` interface.
- `POST /control` accepts web control and `status.json` reflects `control_source=web` during repeated publications.
- `POST /control/neutral` returns control to neutral by watchdog.
- Jetson inference endpoint: `GET http://100.115.99.8:9001/info` responds `Roboflow Inference Server 1.1.2`.

Current real-frame state:

```json
{
  "control_enabled": true,
  "has_video": false,
  "last_client": "172.16.0.2",
  "packet_types": {
    "B": 114
  },
  "video_frames": 0,
  "web_port": 8088
}
```

`AM-Cloud` was republished on `1/command`; for the following 30 seconds EPC received only `B` packets, not `I` packets. Therefore, the EPC interface and runtime are active, but the car is not sending camera frames to the UDP endpoint.

Car access:

- `172.16.0.2:22` is open.
- Non-interactive SSH failed for known users `tp2`, `grupo4`, `pi`, `ubuntu`, `artemis`.
- Without car credentials it was not possible to log in and inspect or start the camera process.
