from __future__ import annotations

import json
import os
import pickle
import signal
import socket
import struct
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

os.environ.setdefault("TP2_INFERENCE_MODE", "local")
os.environ.setdefault("TP2_INFERENCE_TARGET", "model")
os.environ.setdefault("ROBOFLOW_LOCAL_API_URL", "http://100.115.99.8:9001")
os.environ.setdefault("ROBOFLOW_MODEL_ID", "tp2-g4-2026/2")

from roboflow_runtime import (  # noqa: E402
    InferenceConfig,
    create_client,
    draw_predictions_on_image,
    extract_predictions,
    infer_one_image,
    local_endpoint_reachable,
)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


BIND_IP = os.getenv("TP2_BIND_IP", "172.16.0.1")
BIND_PORT = env_int("TP2_BIND_PORT", 20001)
UDP_RECV_BYTES = env_int("TP2_UDP_RECV_BYTES", 131072)

WEB_HOST = os.getenv("TP2_WEB_HOST", "0.0.0.0")
WEB_PORT = env_int("TP2_WEB_PORT", 8088)
ENABLE_WEB_VIEW = env_bool("TP2_ENABLE_WEB_VIEW", True)
ENABLE_WEB_CONTROL = env_bool("TP2_ENABLE_WEB_CONTROL", True)
ENABLE_INFERENCE = env_bool("TP2_ENABLE_INFERENCE", True)

NEUTRAL_STEERING = env_float("TP2_NEUTRAL_STEERING", 0.25)
NEUTRAL_THROTTLE = env_float("TP2_NEUTRAL_THROTTLE", 0.0)
CONTROL_TIMEOUT_SEC = env_float("TP2_WEB_CONTROL_TIMEOUT_SEC", 0.45)
CONTROL_TX_HZ = max(1.0, env_float("TP2_CONTROL_TX_HZ", 20.0))
CLIENT_ADDR_TTL_SEC = env_float("TP2_CLIENT_ADDR_TTL_SEC", 3.0)

INFERENCE_MIN_INTERVAL_SEC = env_float("TP2_INFERENCE_MIN_INTERVAL_SEC", 0.18)
INFERENCE_RETRY_SEC = env_float("TP2_INFERENCE_RETRY_SEC", 2.0)
INFERENCE_MIN_CONFIDENCE = env_float("TP2_INFERENCE_MIN_CONFIDENCE", 0.20)
OVERLAY_MAX_AGE_SEC = env_float("TP2_OVERLAY_MAX_AGE_SEC", 1.25)
JPEG_QUALITY = min(95, max(35, env_int("TP2_JPEG_QUALITY", 78)))

EXIT_EVENT = threading.Event()


def clamp(value: Any, minimum: float, maximum: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def monotonic_ms() -> int:
    return int(time.monotonic() * 1000)


def wall_time() -> float:
    return time.time()


@dataclass
class FrameContext:
    frame: np.ndarray | None
    seq: int
    frame_time: float | None
    predictions: list[dict[str, Any]]
    predictions_time: float | None
    inference_status: str
    inference_latency_ms: int | None


class RuntimeState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.frame_cond = threading.Condition(self.lock)

        self.started_at = wall_time()
        self.packets: Counter[str] = Counter()
        self.bad_packets = 0
        self.tx_packets = 0
        self.last_packet_at: float | None = None
        self.last_packet_type: str | None = None
        self.last_packet_error: str | None = None
        self.last_client_addr: tuple[str, int] | None = None

        self.latest_frame: np.ndarray | None = None
        self.latest_frame_seq = 0
        self.latest_frame_at: float | None = None
        self.frame_decode_errors = 0

        self.battery: float | None = None
        self.telemetry: Any = None

        self.predictions: list[dict[str, Any]] = []
        self.predictions_seq = 0
        self.predictions_at: float | None = None
        self.inference_status = "disabled" if not ENABLE_INFERENCE else "starting"
        self.inference_error: str | None = None
        self.inference_latency_ms: int | None = None
        self.inference_backend: dict[str, Any] = {}
        self.inference_frames = 0

        self.control_armed = False
        self.control_source = "neutral"
        self.steering = NEUTRAL_STEERING
        self.throttle = NEUTRAL_THROTTLE
        self.control_updated_at = wall_time()
        self.control_seq = 0

        self.web_stream_clients = 0
        self.web_control_posts = 0

    def note_packet(
        self,
        packet_type: str,
        address: tuple[str, int],
        *,
        error: str | None = None,
    ) -> None:
        with self.lock:
            now = wall_time()
            self.packets[packet_type] += 1
            self.last_packet_at = now
            self.last_packet_type = packet_type
            self.last_client_addr = address
            if error:
                self.bad_packets += 1
                self.last_packet_error = error[:240]

    def note_tx(self) -> None:
        with self.lock:
            self.tx_packets += 1

    def update_battery(self, value: Any) -> None:
        with self.lock:
            try:
                self.battery = float(value)
            except (TypeError, ValueError):
                self.telemetry = summarize_payload(value)

    def update_telemetry(self, value: Any) -> None:
        with self.lock:
            self.telemetry = summarize_payload(value)

    def update_frame(self, frame: np.ndarray) -> int:
        with self.frame_cond:
            self.latest_frame = frame
            self.latest_frame_seq += 1
            self.latest_frame_at = wall_time()
            seq = self.latest_frame_seq
            self.frame_cond.notify_all()
            return seq

    def note_frame_decode_error(self, message: str) -> None:
        with self.lock:
            self.frame_decode_errors += 1
            self.last_packet_error = message[:240]

    def frame_context(self) -> FrameContext:
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            return FrameContext(
                frame=frame,
                seq=self.latest_frame_seq,
                frame_time=self.latest_frame_at,
                predictions=list(self.predictions),
                predictions_time=self.predictions_at,
                inference_status=self.inference_status,
                inference_latency_ms=self.inference_latency_ms,
            )

    def wait_for_frame(self, last_seq: int, timeout: float) -> FrameContext:
        with self.frame_cond:
            deadline = time.monotonic() + timeout
            while self.latest_frame_seq == last_seq and not EXIT_EVENT.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.frame_cond.wait(remaining)
        return self.frame_context()

    def set_inference_backend(self, config: InferenceConfig) -> None:
        with self.lock:
            self.inference_backend = {
                "mode": config.mode,
                "target": config.target,
                "api_url": config.api_url,
                "model_id": config.model_id,
            }

    def set_inference_status(self, status: str, error: str | None = None) -> None:
        with self.lock:
            self.inference_status = status
            self.inference_error = error[:300] if error else None

    def set_predictions(
        self,
        seq: int,
        predictions: list[dict[str, Any]],
        latency_ms: int,
    ) -> None:
        with self.lock:
            self.predictions = predictions
            self.predictions_seq = seq
            self.predictions_at = wall_time()
            self.inference_status = "ready"
            self.inference_error = None
            self.inference_latency_ms = latency_ms
            self.inference_frames += 1

    def _apply_control_watchdog_locked(self) -> None:
        if (
            self.control_source == "web"
            and wall_time() - self.control_updated_at > CONTROL_TIMEOUT_SEC
        ):
            self.control_armed = False
            self.control_source = "watchdog"
            self.steering = NEUTRAL_STEERING
            self.throttle = NEUTRAL_THROTTLE
            self.control_updated_at = wall_time()
            self.control_seq += 1

    def set_control(
        self,
        steering: Any,
        throttle: Any,
        *,
        source: str,
        armed: bool,
    ) -> dict[str, Any]:
        with self.lock:
            self.web_control_posts += 1
            if not ENABLE_WEB_CONTROL or not armed:
                self.control_armed = False
                self.control_source = "neutral"
                self.steering = NEUTRAL_STEERING
                self.throttle = NEUTRAL_THROTTLE
            else:
                self.control_armed = True
                self.control_source = source
                self.steering = round(clamp(steering, -1.0, 1.0, NEUTRAL_STEERING), 3)
                self.throttle = round(clamp(throttle, -1.0, 1.0, NEUTRAL_THROTTLE), 3)
            self.control_updated_at = wall_time()
            self.control_seq += 1
            return self.control_snapshot_locked()

    def neutral(self, source: str = "neutral") -> dict[str, Any]:
        with self.lock:
            self.control_armed = False
            self.control_source = source
            self.steering = NEUTRAL_STEERING
            self.throttle = NEUTRAL_THROTTLE
            self.control_updated_at = wall_time()
            self.control_seq += 1
            return self.control_snapshot_locked()

    def control_snapshot_locked(self) -> dict[str, Any]:
        return {
            "armed": self.control_armed,
            "source": self.control_source,
            "steering": self.steering,
            "throttle": self.throttle,
            "updated_age_sec": max(0.0, wall_time() - self.control_updated_at),
            "seq": self.control_seq,
        }

    def get_control(self) -> tuple[float, float, dict[str, Any]]:
        with self.lock:
            self._apply_control_watchdog_locked()
            return self.steering, self.throttle, self.control_snapshot_locked()

    def get_client_address(self) -> tuple[str, int] | None:
        with self.lock:
            if self.last_client_addr is None or self.last_packet_at is None:
                return None
            if wall_time() - self.last_packet_at > CLIENT_ADDR_TTL_SEC:
                return None
            return self.last_client_addr

    def add_stream_client(self) -> None:
        with self.lock:
            self.web_stream_clients += 1

    def remove_stream_client(self) -> None:
        with self.lock:
            self.web_stream_clients = max(0, self.web_stream_clients - 1)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self._apply_control_watchdog_locked()
            now = wall_time()
            has_video = self.latest_frame is not None
            inference_age = (
                None if self.predictions_at is None else max(0.0, now - self.predictions_at)
            )
            video_age = None if self.latest_frame_at is None else max(0.0, now - self.latest_frame_at)
            packet_age = None if self.last_packet_at is None else max(0.0, now - self.last_packet_at)
            return {
                "ok": True,
                "uptime_sec": round(now - self.started_at, 3),
                "udp": {
                    "bind": f"{BIND_IP}:{BIND_PORT}",
                    "last_client": format_address(self.last_client_addr),
                    "last_packet_type": self.last_packet_type,
                    "last_packet_age_sec": rounded(packet_age),
                    "packets": dict(self.packets),
                    "bad_packets": self.bad_packets,
                    "tx_packets": self.tx_packets,
                    "last_error": self.last_packet_error,
                },
                "video": {
                    "has_video": has_video,
                    "frames": self.latest_frame_seq,
                    "age_sec": rounded(video_age),
                    "decode_errors": self.frame_decode_errors,
                },
                "inference": {
                    "enabled": ENABLE_INFERENCE,
                    "status": self.inference_status,
                    "error": self.inference_error,
                    "latency_ms": self.inference_latency_ms,
                    "age_sec": rounded(inference_age),
                    "frames": self.inference_frames,
                    "detections": len(self.predictions),
                    "predictions": sanitize_predictions(self.predictions),
                    "backend": self.inference_backend,
                },
                "control": self.control_snapshot_locked(),
                "car": {
                    "battery": self.battery,
                    "telemetry": self.telemetry,
                },
                "web": {
                    "host": WEB_HOST,
                    "port": WEB_PORT,
                    "control_enabled": ENABLE_WEB_CONTROL,
                    "stream_clients": self.web_stream_clients,
                    "control_posts": self.web_control_posts,
                },
            }


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def format_address(address: tuple[str, int] | None) -> str | None:
    if address is None:
        return None
    return f"{address[0]}:{address[1]}"


def summarize_payload(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "len": len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): summarize_payload(v) for k, v in list(value.items())[:16]}
    if isinstance(value, (list, tuple)):
        return [summarize_payload(item) for item in list(value)[:16]]
    text = repr(value)
    return text[:400]


def sanitize_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for prediction in predictions[:20]:
        item: dict[str, Any] = {}
        for key in ("class", "confidence", "x", "y", "width", "height"):
            value = prediction.get(key)
            if isinstance(value, (int, float)):
                item[key] = round(float(value), 4)
            elif value is not None:
                item[key] = str(value)
        clean.append(item)
    return clean


def decode_pickle_payload(payload: bytes) -> Any:
    return pickle.loads(payload, encoding="latin1")


def decode_image_payload(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        if value.dtype != np.uint8:
            value = value.astype(np.uint8)
        if value.ndim == 1:
            return cv2.imdecode(value, cv2.IMREAD_COLOR)
        if value.ndim in {2, 3}:
            return value.copy()
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = np.frombuffer(bytes(value), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if isinstance(value, (list, tuple)):
        data = np.asarray(value, dtype=np.uint8)
        if data.ndim == 1:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if data.ndim in {2, 3}:
            return data.copy()
    if isinstance(value, dict):
        for key in ("image", "frame", "jpg", "jpeg", "data"):
            if key in value:
                frame = decode_image_payload(value[key])
                if frame is not None:
                    return frame
    return None


def parse_car_packet(packet: bytes) -> tuple[str, Any]:
    if not packet:
        raise ValueError("empty packet")
    packet_type = chr(packet[0])
    payload = packet[1:]
    if packet_type == "I":
        try:
            return packet_type, decode_pickle_payload(payload)
        except Exception:
            return packet_type, payload
    if not payload:
        return packet_type, None
    return packet_type, decode_pickle_payload(payload)


def send_control_packet(
    sock: socket.socket,
    address: tuple[str, int],
    steering: float,
    throttle: float,
) -> None:
    payload = (
        struct.pack("c", b"C")
        + struct.pack("d", round(float(steering), 3))
        + struct.pack("d", round(float(throttle), 3))
    )
    sock.sendto(payload, address)


def encode_jpeg(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return encoded.tobytes()


def draw_status_overlay(
    frame: np.ndarray,
    context: FrameContext,
    state_snapshot: dict[str, Any],
) -> np.ndarray:
    output = frame.copy()
    h, w = output.shape[:2]
    compact = w < 520 or h < 320
    panel_w = min(w - 16, 520 if not compact else w - 16)
    panel_h = 76 if not compact else 42
    x0 = 12 if not compact else 8
    y0 = 12 if not compact else 8
    overlay = output.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (5, 9, 11), -1)
    cv2.addWeighted(overlay, 0.68, output, 0.32, 0, output)

    inf = state_snapshot["inference"]
    udp = state_snapshot["udp"]
    det = inf["detections"]
    latency = inf["latency_ms"]
    latency_text = "-" if latency is None else f"{latency}ms"
    if compact:
        lines = [f"f {context.seq}  det {det}  ia {inf['status']}"]
        scale = 0.42
        y = y0 + 26
    else:
        lines = [
            f"frame {context.seq}  det {det}  ia {inf['status']}  {latency_text}",
            f"rx {udp['packets']}  tx {udp['tx_packets']}  cliente {udp['last_client'] or '-'}",
        ]
        scale = 0.56
        y = y0 + 30
    for line in lines:
        cv2.putText(
            output,
            line,
            (x0 + 14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (235, 244, 239),
            1,
            cv2.LINE_AA,
        )
        y += 24
    return output


def build_stream_frame(state: RuntimeState) -> bytes:
    context = state.frame_context()
    snapshot = state.snapshot()
    if context.frame is None:
        return encode_jpeg(build_placeholder(snapshot))

    frame = context.frame
    predictions_are_current = (
        context.predictions
        and context.predictions_time is not None
        and wall_time() - context.predictions_time <= OVERLAY_MAX_AGE_SEC
    )
    if predictions_are_current:
        frame = draw_predictions_on_image(
            frame,
            context.predictions,
            min_confidence=INFERENCE_MIN_CONFIDENCE,
        )

    frame = draw_status_overlay(frame, context, snapshot)
    return encode_jpeg(frame)


def build_placeholder(snapshot: dict[str, Any]) -> np.ndarray:
    width, height = 1280, 720
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (12, 14, 15)

    for x in range(0, width, 80):
        cv2.line(canvas, (x, 0), (x, height), (22, 26, 28), 1)
    for y in range(0, height, 80):
        cv2.line(canvas, (0, y), (width, y), (22, 26, 28), 1)

    udp = snapshot["udp"]
    video = snapshot["video"]
    inf = snapshot["inference"]
    lines = [
        "SIN FRAME DE CAMARA",
        f"escuchando UDP {udp['bind']}",
        f"cliente {udp['last_client'] or '-'}  ultimo paquete {udp['last_packet_type'] or '-'}",
        f"paquetes {udp['packets']}  frames {video['frames']}",
        f"inferencia {inf['status']}  detecciones {inf['detections']}",
    ]
    y = 250
    for idx, line in enumerate(lines):
        scale = 1.15 if idx == 0 else 0.72
        color = (238, 244, 240) if idx == 0 else (158, 172, 165)
        thickness = 2 if idx == 0 else 1
        cv2.putText(
            canvas,
            line,
            (80, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y += 54
    return canvas


def inference_loop(state: RuntimeState) -> None:
    last_seq = 0
    last_submit = 0.0

    while not EXIT_EVENT.is_set():
        try:
            config = InferenceConfig.from_env()
            config.validate()
            state.set_inference_backend(config)

            if config.mode == "local" and not local_endpoint_reachable(config.api_url, 2.0):
                state.set_inference_status("offline", f"no reach {config.api_url}")
                EXIT_EVENT.wait(INFERENCE_RETRY_SEC)
                continue

            client = create_client(config)
            state.set_inference_status("waiting-frame")

            while not EXIT_EVENT.is_set():
                context = state.wait_for_frame(last_seq, timeout=1.0)
                if context.frame is None or context.seq == last_seq:
                    continue

                sleep_for = INFERENCE_MIN_INTERVAL_SEC - (time.monotonic() - last_submit)
                if sleep_for > 0:
                    EXIT_EVENT.wait(sleep_for)
                if EXIT_EVENT.is_set():
                    break

                frame = context.frame
                seq = context.seq
                last_seq = seq
                last_submit = time.monotonic()
                state.set_inference_status("running")

                temp_path: Path | None = None
                started_ms = monotonic_ms()
                try:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        temp_path = Path(tmp.name)
                    if not cv2.imwrite(str(temp_path), frame):
                        raise RuntimeError("could not write temp inference frame")
                    payload = infer_one_image(client, temp_path, config)
                    predictions = extract_predictions(payload)
                    latency = monotonic_ms() - started_ms
                    state.set_predictions(seq, predictions, latency)
                finally:
                    if temp_path is not None:
                        try:
                            temp_path.unlink(missing_ok=True)
                        except OSError:
                            pass

        except Exception as exc:
            state.set_inference_status("error", str(exc))
            EXIT_EVENT.wait(INFERENCE_RETRY_SEC)


def control_tx_loop(sock: socket.socket, state: RuntimeState) -> None:
    interval = 1.0 / CONTROL_TX_HZ
    while not EXIT_EVENT.wait(interval):
        address = state.get_client_address()
        if address is None:
            continue
        steering, throttle, _ = state.get_control()
        try:
            send_control_packet(sock, address, steering, throttle)
            state.note_tx()
        except OSError as exc:
            state.note_packet("TX_ERROR", address, error=str(exc))


class LiveHandler(BaseHTTPRequestHandler):
    state: RuntimeState

    server_version = "TP2Live/2.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(
            f"{self.address_string()} - - [{self.log_date_time_string()}] {fmt % args}\n"
        )

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_html(LIVE_VIEW_HTML)
        elif path == "/status.json":
            self.send_json(self.state.snapshot())
        elif path == "/snapshot.jpg":
            self.send_image(build_stream_frame(self.state))
        elif path == "/video.mjpg":
            self.stream_video()
        elif path == "/healthz":
            self.send_json({"ok": True})
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
        else:
            self.send_error(404, "not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path in {"/control/neutral", "/neutral"}:
            self.send_json({"ok": True, "control": self.state.neutral("neutral")})
            return
        if path != "/control":
            self.send_error(404, "not found")
            return
        if not ENABLE_WEB_CONTROL:
            self.send_json({"ok": False, "error": "web control disabled"}, status=403)
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(min(length, 8192)) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_json({"ok": False, "error": "invalid json"}, status=400)
            return

        action = str(payload.get("action", "")).strip().lower()
        if action in {"neutral", "stop", "estop"}:
            control = self.state.neutral("stop" if action != "neutral" else "neutral")
        else:
            control = self.state.set_control(
                payload.get("steering", NEUTRAL_STEERING),
                payload.get("throttle", NEUTRAL_THROTTLE),
                source="web",
                armed=bool(payload.get("armed", True)),
            )
        self.send_json({"ok": True, "control": control})

    def send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_image(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def stream_video(self) -> None:
        self.state.add_stream_client()
        boundary = b"tp2frame"
        last_seq = 0
        try:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary.decode()}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.end_headers()

            while not EXIT_EVENT.is_set():
                context = self.state.wait_for_frame(last_seq, timeout=1.0)
                last_seq = context.seq
                frame = build_stream_frame(self.state)
                header = (
                    b"--"
                    + boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode("ascii")
                    + b"\r\n\r\n"
                )
                self.wfile.write(header)
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            self.state.remove_stream_client()


def start_http_server(state: RuntimeState) -> ThreadingHTTPServer | None:
    if not ENABLE_WEB_VIEW:
        return None
    LiveHandler.state = state
    server = ThreadingHTTPServer((WEB_HOST, WEB_PORT), LiveHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="web")
    thread.start()
    print(f"Live web view listening on http://{WEB_HOST}:{WEB_PORT}/", flush=True)
    return server


def handle_udp_packet(
    packet: bytes,
    address: tuple[str, int],
    sock: socket.socket,
    state: RuntimeState,
) -> None:
    try:
        packet_type, payload = parse_car_packet(packet)
    except Exception as exc:
        state.note_packet("?", address, error=f"parse: {exc}")
        return

    state.note_packet(packet_type, address)

    if packet_type == "I":
        frame = decode_image_payload(payload)
        if frame is None:
            state.note_frame_decode_error("could not decode image packet")
        else:
            state.update_frame(frame)
    elif packet_type == "B":
        state.update_battery(payload)
    elif packet_type == "D":
        state.update_telemetry(payload)
    else:
        state.note_packet(packet_type, address, error="unknown packet type")

    steering, throttle, _ = state.get_control()
    try:
        send_control_packet(sock, address, steering, throttle)
        state.note_tx()
    except OSError as exc:
        state.note_packet("TX_ERROR", address, error=str(exc))


def install_signal_handlers(server: ThreadingHTTPServer | None, sock: socket.socket) -> None:
    def stop(_signum: int, _frame: Any) -> None:
        EXIT_EVENT.set()
        try:
            sock.close()
        except OSError:
            pass
        if server is not None:
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)


LIVE_VIEW_HTML = r"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TP2 Live Control</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b0d0e;
      --panel: #141719;
      --panel-2: #1a1e20;
      --line: #2d3438;
      --text: #eef3ef;
      --muted: #98a39d;
      --green: #46d482;
      --amber: #f2be4b;
      --red: #ff5f57;
      --cyan: #62c7d9;
      --shadow: rgba(0, 0, 0, 0.28);
    }

    * { box-sizing: border-box; }

    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      background:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px),
        var(--bg);
      background-size: 40px 40px;
      color: var(--text);
      font: 14px/1.4 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }

    button {
      height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #202529;
      color: var(--text);
      padding: 0 14px;
      font-weight: 800;
      letter-spacing: 0;
      cursor: pointer;
    }

    button:hover { border-color: #526068; }
    button:active { transform: translateY(1px); }
    button.primary { background: #1d6f47; border-color: #2b9c66; }
    button.danger { background: #6f2525; border-color: #b94842; }

    .app {
      height: 100%;
      display: grid;
      grid-template-rows: 74px 1fr;
      gap: 14px;
      padding: 18px;
    }

    header {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      align-items: end;
      gap: 18px;
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 52px);
      line-height: 0.95;
      letter-spacing: 0;
      font-weight: 900;
    }

    .subtitle {
      color: var(--muted);
      margin-top: 8px;
      font-weight: 650;
    }

    .status-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 34px;
      padding: 0 11px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(20, 23, 25, 0.82);
      box-shadow: 0 12px 28px var(--shadow);
      color: var(--muted);
      white-space: nowrap;
      font-weight: 800;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 99px;
      background: var(--muted);
      box-shadow: 0 0 16px currentColor;
    }

    .ok .dot { background: var(--green); color: var(--green); }
    .warn .dot { background: var(--amber); color: var(--amber); }
    .bad .dot { background: var(--red); color: var(--red); }

    main {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
    }

    .video-shell, .side-panel, .control-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(20, 23, 25, 0.9);
      box-shadow: 0 24px 60px rgba(0,0,0,0.32);
    }

    .video-shell {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: 1fr 112px;
      overflow: hidden;
    }

    .video-frame {
      position: relative;
      min-height: 0;
      background: #030404;
      display: grid;
      place-items: center;
    }

    .video-frame img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    .video-badge {
      position: absolute;
      left: 18px;
      bottom: 18px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .badge {
      background: rgba(5, 8, 8, 0.74);
      border: 1px solid rgba(255,255,255,0.13);
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--text);
      font-weight: 900;
    }

    .control-panel {
      border-width: 1px 0 0 0;
      border-radius: 0;
      display: grid;
      grid-template-columns: auto auto 1fr;
      align-items: center;
      gap: 14px;
      padding: 16px;
      background: var(--panel-2);
    }

    .axis-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      min-width: 0;
    }

    .axis label {
      display: flex;
      justify-content: space-between;
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
      margin-bottom: 6px;
    }

    .bar {
      height: 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #0b0e0f;
      overflow: hidden;
    }

    .fill {
      height: 100%;
      width: 50%;
      background: linear-gradient(90deg, var(--cyan), var(--green));
      transform-origin: left;
    }

    .side-panel {
      min-height: 0;
      overflow: auto;
      padding: 16px;
      display: grid;
      align-content: start;
      gap: 12px;
    }

    .section {
      border-top: 1px solid var(--line);
      padding-top: 13px;
    }

    .section:first-child {
      border-top: 0;
      padding-top: 0;
    }

    .section-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 950;
      text-transform: uppercase;
      margin-bottom: 10px;
    }

    .metric {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      padding: 7px 0;
      border-bottom: 1px solid rgba(255,255,255,0.055);
    }

    .metric span:first-child {
      color: var(--muted);
      font-weight: 700;
    }

    .metric span:last-child {
      font-variant-numeric: tabular-nums;
      font-weight: 900;
      text-align: right;
    }

    .detections {
      display: grid;
      gap: 6px;
      max-height: 160px;
      overflow: auto;
    }

    .det {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 8px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #101315;
      font-weight: 800;
    }

    .raw {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      color: #74e89a;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      max-height: 210px;
      overflow: auto;
    }

    @media (max-width: 980px) {
      html, body { overflow: auto; }
      .app { height: auto; min-height: 100%; }
      header { grid-template-columns: 1fr; align-items: start; }
      .status-row { justify-content: flex-start; }
      main { grid-template-columns: 1fr; }
      .video-shell { min-height: 62vh; }
      .control-panel { grid-template-columns: 1fr; }
      .axis-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>TP2 Live Control</h1>
        <div class="subtitle">EPC gateway · Jetson inference · browser control</div>
      </div>
      <div class="status-row">
        <div class="pill warn" id="pill-video"><span class="dot"></span><span>VIDEO</span><strong id="top-video">--</strong></div>
        <div class="pill warn" id="pill-ai"><span class="dot"></span><span>IA</span><strong id="top-ai">--</strong></div>
        <div class="pill warn" id="pill-car"><span class="dot"></span><span>COCHE</span><strong id="top-car">--</strong></div>
        <div class="pill bad" id="pill-control"><span class="dot"></span><span>CTRL</span><strong id="top-control">OFF</strong></div>
      </div>
    </header>

    <main>
      <section class="video-shell">
        <div class="video-frame">
          <img id="video" src="/video.mjpg" alt="TP2 live video">
          <div class="video-badge">
            <div class="badge" id="badge-frame">frame --</div>
            <div class="badge" id="badge-latency">lat --</div>
            <div class="badge" id="badge-det">det --</div>
          </div>
        </div>
        <div class="control-panel">
          <button class="primary" id="arm">ARMAR</button>
          <button class="danger" id="neutral">NEUTRO</button>
          <div class="axis-grid">
            <div class="axis">
              <label><span>Giro</span><strong id="steer-value">0.25</strong></label>
              <div class="bar"><div class="fill" id="steer-fill"></div></div>
            </div>
            <div class="axis">
              <label><span>Gas</span><strong id="throttle-value">0.00</strong></label>
              <div class="bar"><div class="fill" id="throttle-fill"></div></div>
            </div>
          </div>
        </div>
      </section>

      <aside class="side-panel">
        <section class="section">
          <div class="section-title">Enlace</div>
          <div class="metric"><span>UDP</span><span id="udp-bind">--</span></div>
          <div class="metric"><span>Cliente</span><span id="client">--</span></div>
          <div class="metric"><span>Ultimo paquete</span><span id="last-packet">--</span></div>
          <div class="metric"><span>RX</span><span id="packets">--</span></div>
          <div class="metric"><span>TX</span><span id="tx">--</span></div>
        </section>

        <section class="section">
          <div class="section-title">Inferencia</div>
          <div class="metric"><span>Estado</span><span id="ai-status">--</span></div>
          <div class="metric"><span>Backend</span><span id="backend">--</span></div>
          <div class="metric"><span>Latencia</span><span id="latency">--</span></div>
          <div class="detections" id="detections"></div>
        </section>

        <section class="section">
          <div class="section-title">Coche</div>
          <div class="metric"><span>Bateria</span><span id="battery">--</span></div>
          <div class="metric"><span>Control</span><span id="control-source">--</span></div>
          <div class="metric"><span>Watchdog</span><span id="watchdog">--</span></div>
        </section>

        <section class="section">
          <div class="section-title">Estado</div>
          <pre class="raw" id="raw">{}</pre>
        </section>
      </aside>
    </main>
  </div>

  <script>
    const els = {
      arm: document.getElementById('arm'),
      neutral: document.getElementById('neutral'),
      steerValue: document.getElementById('steer-value'),
      throttleValue: document.getElementById('throttle-value'),
      steerFill: document.getElementById('steer-fill'),
      throttleFill: document.getElementById('throttle-fill'),
      topVideo: document.getElementById('top-video'),
      topAi: document.getElementById('top-ai'),
      topCar: document.getElementById('top-car'),
      topControl: document.getElementById('top-control'),
      pillVideo: document.getElementById('pill-video'),
      pillAi: document.getElementById('pill-ai'),
      pillCar: document.getElementById('pill-car'),
      pillControl: document.getElementById('pill-control'),
      frame: document.getElementById('badge-frame'),
      badgeLatency: document.getElementById('badge-latency'),
      badgeDet: document.getElementById('badge-det'),
      udpBind: document.getElementById('udp-bind'),
      client: document.getElementById('client'),
      lastPacket: document.getElementById('last-packet'),
      packets: document.getElementById('packets'),
      tx: document.getElementById('tx'),
      aiStatus: document.getElementById('ai-status'),
      backend: document.getElementById('backend'),
      latency: document.getElementById('latency'),
      detections: document.getElementById('detections'),
      battery: document.getElementById('battery'),
      controlSource: document.getElementById('control-source'),
      watchdog: document.getElementById('watchdog'),
      raw: document.getElementById('raw'),
    };

    let armed = false;
    let keys = new Set();
    let lastControl = {steering: 0.25, throttle: 0.0};

    function setPill(el, state) {
      el.classList.remove('ok', 'warn', 'bad');
      el.classList.add(state);
    }

    function axisFromKeys() {
      let throttle = 0.0;
      if (keys.has('w') || keys.has('arrowup')) throttle = 0.6;
      if (keys.has('s') || keys.has('arrowdown')) throttle = -0.5;
      if (keys.has('x') || keys.has(' ')) throttle = -0.9;

      let steering = 0.25;
      const left = keys.has('a') || keys.has('arrowleft');
      const right = keys.has('d') || keys.has('arrowright');
      if (left && !right) steering = 1.0;
      if (right && !left) steering = -1.0;
      return {steering, throttle};
    }

    function renderAxis(control) {
      els.steerValue.textContent = Number(control.steering).toFixed(2);
      els.throttleValue.textContent = Number(control.throttle).toFixed(2);
      els.steerFill.style.width = `${((Number(control.steering) + 1) / 2) * 100}%`;
      els.throttleFill.style.width = `${((Number(control.throttle) + 1) / 2) * 100}%`;
    }

    async function postControl(control) {
      try {
        const res = await fetch('/control', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({...control, armed}),
          cache: 'no-store',
        });
        if (!res.ok) throw new Error(`http ${res.status}`);
      } catch (err) {
        setPill(els.pillControl, 'bad');
      }
    }

    async function neutral() {
      armed = false;
      keys.clear();
      els.arm.textContent = 'ARMAR';
      setPill(els.pillControl, 'bad');
      lastControl = {steering: 0.25, throttle: 0.0};
      renderAxis(lastControl);
      try {
        await fetch('/control/neutral', {method: 'POST', cache: 'no-store'});
      } catch (_) {}
    }

    els.arm.addEventListener('click', () => {
      armed = !armed;
      els.arm.textContent = armed ? 'ARMADO' : 'ARMAR';
      setPill(els.pillControl, armed ? 'ok' : 'bad');
      document.body.focus();
      if (!armed) neutral();
    });

    els.neutral.addEventListener('click', neutral);

    window.addEventListener('keydown', (event) => {
      const key = event.key.toLowerCase();
      if (['w','a','s','d','x',' ','arrowup','arrowdown','arrowleft','arrowright'].includes(key)) {
        if (armed) event.preventDefault();
        keys.add(key);
      }
    });

    window.addEventListener('keyup', (event) => {
      keys.delete(event.key.toLowerCase());
    });

    window.addEventListener('blur', neutral);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) neutral();
    });

    setInterval(() => {
      if (!armed) return;
      lastControl = axisFromKeys();
      renderAxis(lastControl);
      postControl(lastControl);
    }, 50);

    async function pollStatus() {
      try {
        const res = await fetch('/status.json', {cache: 'no-store'});
        const data = await res.json();

        const videoAge = data.video.age_sec;
        const videoOk = data.video.has_video && (videoAge === null || videoAge < 1.5);
        els.topVideo.textContent = videoOk ? `${data.video.frames}` : 'SIN';
        setPill(els.pillVideo, videoOk ? 'ok' : 'warn');

        const aiOk = ['ready', 'running', 'waiting-frame'].includes(data.inference.status);
        els.topAi.textContent = data.inference.status;
        setPill(els.pillAi, aiOk ? 'ok' : 'warn');

        const carOk = data.udp.last_packet_age_sec !== null && data.udp.last_packet_age_sec < 3.0;
        els.topCar.textContent = carOk ? 'ONLINE' : 'SIN RX';
        setPill(els.pillCar, carOk ? 'ok' : 'warn');

        els.topControl.textContent = data.control.armed ? 'ON' : 'OFF';
        setPill(els.pillControl, data.control.armed ? 'ok' : 'bad');

        els.frame.textContent = `frame ${data.video.frames}`;
        els.badgeLatency.textContent = `lat ${data.inference.latency_ms ?? '--'}ms`;
        els.badgeDet.textContent = `det ${data.inference.detections}`;

        els.udpBind.textContent = data.udp.bind;
        els.client.textContent = data.udp.last_client || '--';
        els.lastPacket.textContent = `${data.udp.last_packet_type || '--'} · ${data.udp.last_packet_age_sec ?? '--'}s`;
        els.packets.textContent = JSON.stringify(data.udp.packets);
        els.tx.textContent = `${data.udp.tx_packets}`;

        els.aiStatus.textContent = data.inference.error || data.inference.status;
        els.backend.textContent = data.inference.backend.api_url || '--';
        els.latency.textContent = data.inference.latency_ms === null ? '--' : `${data.inference.latency_ms} ms`;

        els.detections.innerHTML = '';
        const preds = data.inference.predictions || [];
        if (preds.length === 0) {
          const empty = document.createElement('div');
          empty.className = 'det';
          empty.innerHTML = '<span>sin detecciones</span><span>--</span>';
          els.detections.appendChild(empty);
        } else {
          for (const pred of preds) {
            const row = document.createElement('div');
            row.className = 'det';
            const conf = pred.confidence === undefined ? '--' : Number(pred.confidence).toFixed(2);
            row.innerHTML = `<span>${pred.class || 'objeto'}</span><span>${conf}</span>`;
            els.detections.appendChild(row);
          }
        }

        els.battery.textContent = data.car.battery === null ? '--' : Number(data.car.battery).toFixed(2);
        els.controlSource.textContent = `${data.control.source} · ${Number(data.control.steering).toFixed(2)} / ${Number(data.control.throttle).toFixed(2)}`;
        els.watchdog.textContent = `${Number(data.control.updated_age_sec).toFixed(2)}s`;
        renderAxis(data.control.armed ? lastControl : data.control);
        els.raw.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        setPill(els.pillCar, 'bad');
        setPill(els.pillVideo, 'bad');
      }
    }

    pollStatus();
    setInterval(pollStatus, 500);
  </script>
</body>
</html>
"""


def main() -> int:
    state = RuntimeState()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.5)

    try:
        sock.bind((BIND_IP, BIND_PORT))
    except OSError as exc:
        print(f"Could not bind UDP {BIND_IP}:{BIND_PORT}: {exc}", file=sys.stderr, flush=True)
        return 2

    server = start_http_server(state)
    install_signal_handlers(server, sock)

    if ENABLE_INFERENCE:
        threading.Thread(target=inference_loop, args=(state,), daemon=True, name="inference").start()
    threading.Thread(target=control_tx_loop, args=(sock, state), daemon=True, name="control-tx").start()

    print(
        f"TP2 car runtime listening on UDP {BIND_IP}:{BIND_PORT}; "
        f"web={ENABLE_WEB_VIEW} inference={ENABLE_INFERENCE} control={ENABLE_WEB_CONTROL}",
        flush=True,
    )

    while not EXIT_EVENT.is_set():
        try:
            packet, address = sock.recvfrom(UDP_RECV_BYTES)
        except socket.timeout:
            continue
        except OSError:
            if EXIT_EVENT.is_set():
                break
            raise
        handle_udp_packet(packet, address, sock, state)

    try:
        sock.close()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
