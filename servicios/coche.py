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

from autonomous_driver import (  # noqa: E402
    AutonomousConfig,
    AutonomousDecision,
    decide_autonomous_control,
)
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

DEFAULT_DRIVE_MODE = os.getenv("TP2_DEFAULT_DRIVE_MODE", "manual").strip().lower()
AUTONOMOUS_CONFIG = AutonomousConfig(
    min_confidence=max(
        INFERENCE_MIN_CONFIDENCE,
        env_float("TP2_AUTONOMOUS_MIN_CONFIDENCE", 0.35),
    ),
    stale_prediction_sec=env_float("TP2_AUTONOMOUS_STALE_SEC", 1.25),
    max_frame_age_sec=env_float("TP2_AUTONOMOUS_MAX_FRAME_AGE_SEC", 1.0),
    min_area_ratio=env_float("TP2_AUTONOMOUS_MIN_AREA_RATIO", 0.004),
    near_area_ratio=env_float("TP2_AUTONOMOUS_NEAR_AREA_RATIO", 0.045),
    center_left=env_float("TP2_AUTONOMOUS_CENTER_LEFT", 0.40),
    center_right=env_float("TP2_AUTONOMOUS_CENTER_RIGHT", 0.60),
    neutral_steering=NEUTRAL_STEERING,
    neutral_throttle=NEUTRAL_THROTTLE,
    crawl_throttle=env_float("TP2_AUTONOMOUS_CRAWL_THROTTLE", 0.12),
    slow_throttle=env_float("TP2_AUTONOMOUS_SLOW_THROTTLE", 0.18),
    turn_throttle=env_float("TP2_AUTONOMOUS_TURN_THROTTLE", 0.22),
    cruise_throttle=env_float("TP2_AUTONOMOUS_CRUISE_THROTTLE", 0.34),
    fast_throttle=env_float("TP2_AUTONOMOUS_FAST_THROTTLE", 0.48),
    left_steering=env_float("TP2_AUTONOMOUS_LEFT_STEERING", 0.84),
    right_steering=env_float("TP2_AUTONOMOUS_RIGHT_STEERING", -0.84),
)

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
        self.drive_mode = normalize_drive_mode(DEFAULT_DRIVE_MODE)
        self.autonomous_decision = AutonomousDecision(
            active=False,
            steering=NEUTRAL_STEERING,
            throttle=NEUTRAL_THROTTLE,
            action="safe-neutral",
            reason="not-evaluated",
            target=None,
            candidates=(),
        )
        if self.drive_mode == "autonomous":
            self.control_source = "autonomous"

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

    def _evaluate_autonomous_locked(self) -> AutonomousDecision:
        now = wall_time()
        frame_shape = None if self.latest_frame is None else self.latest_frame.shape
        decision = decide_autonomous_control(
            list(self.predictions),
            frame_shape=frame_shape,
            now=now,
            frame_time=self.latest_frame_at,
            predictions_time=self.predictions_at,
            config=AUTONOMOUS_CONFIG,
        )
        self.autonomous_decision = decision
        return decision

    def _apply_autonomous_control_locked(self) -> AutonomousDecision:
        decision = self._evaluate_autonomous_locked()
        self.control_armed = decision.active
        self.control_source = "autonomous" if decision.active else "autonomous-safe"
        self.steering = decision.steering
        self.throttle = decision.throttle
        self.control_updated_at = wall_time()
        self.control_seq += 1
        return decision

    def set_drive_mode(self, mode: str) -> dict[str, Any]:
        with self.lock:
            self.drive_mode = normalize_drive_mode(mode)
            if self.drive_mode == "autonomous":
                self._apply_autonomous_control_locked()
            else:
                self.control_armed = False
                self.control_source = "mode-manual"
                self.steering = NEUTRAL_STEERING
                self.throttle = NEUTRAL_THROTTLE
                self.control_updated_at = wall_time()
                self.control_seq += 1
            return {
                "mode": self.drive_mode,
                "control": self.control_snapshot_locked(),
                "autonomy": self.autonomous_decision.to_status(),
            }

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
    ) -> dict[str, Any]:
        with self.lock:
            self.web_control_posts += 1
            self.drive_mode = "manual"
            if not ENABLE_WEB_CONTROL:
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
            self.drive_mode = "manual"
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
            "mode": self.drive_mode,
            "steering": self.steering,
            "throttle": self.throttle,
            "updated_age_sec": max(0.0, wall_time() - self.control_updated_at),
            "seq": self.control_seq,
        }

    def get_control(self) -> tuple[float, float, dict[str, Any]]:
        with self.lock:
            if self.drive_mode == "autonomous":
                self._apply_autonomous_control_locked()
                return self.steering, self.throttle, self.control_snapshot_locked()
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
            if self.drive_mode == "autonomous":
                self._apply_autonomous_control_locked()
            else:
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
                "autonomy": {
                    "mode": self.drive_mode,
                    "decision": self.autonomous_decision.to_status(),
                    "config": {
                        "min_confidence": AUTONOMOUS_CONFIG.min_confidence,
                        "stale_prediction_sec": AUTONOMOUS_CONFIG.stale_prediction_sec,
                        "max_frame_age_sec": AUTONOMOUS_CONFIG.max_frame_age_sec,
                        "min_area_ratio": AUTONOMOUS_CONFIG.min_area_ratio,
                        "near_area_ratio": AUTONOMOUS_CONFIG.near_area_ratio,
                    },
                },
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


def normalize_drive_mode(value: str | None) -> str:
    mode = (value or "").strip().lower()
    if mode in {"auto", "autonomous", "autonomo", "autonomous-driving"}:
        return "autonomous"
    return "manual"


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


def normalize_decoded_frame(frame: np.ndarray | None) -> np.ndarray | None:
    if frame is None:
        return None
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()
    return None


def decode_compressed_image(data: np.ndarray) -> np.ndarray | None:
    if data.size == 0:
        return None
    if data.dtype != np.uint8:
        data = data.astype(np.uint8)
    compressed = np.ascontiguousarray(data.reshape(-1))
    return normalize_decoded_frame(cv2.imdecode(compressed, cv2.IMREAD_COLOR))


def decode_image_payload(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        frame = decode_compressed_image(value)
        if frame is not None:
            return frame
        return normalize_decoded_frame(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = np.frombuffer(bytes(value), dtype=np.uint8)
        return decode_compressed_image(data)
    if isinstance(value, (list, tuple)):
        data = np.asarray(value, dtype=np.uint8)
        frame = decode_compressed_image(data)
        if frame is not None:
            return frame
        return normalize_decoded_frame(data)
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
    panel_h = 104 if not compact else 66
    x0 = 12 if not compact else 8
    y0 = 12 if not compact else 8
    overlay = output.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (5, 9, 11), -1)
    cv2.addWeighted(overlay, 0.68, output, 0.32, 0, output)

    inf = state_snapshot["inference"]
    udp = state_snapshot["udp"]
    control = state_snapshot["control"]
    autonomy = state_snapshot.get("autonomy", {}).get("decision", {})
    det = inf["detections"]
    latency = inf["latency_ms"]
    latency_text = "-" if latency is None else f"{latency}ms"
    if compact:
        lines = [
            f"f {context.seq}  det {det}  ia {inf['status']}",
            f"{control['mode']} {control['steering']:.2f}/{control['throttle']:.2f}",
        ]
        scale = 0.42
        y = y0 + 26
    else:
        lines = [
            f"frame {context.seq}  det {det}  ia {inf['status']}  {latency_text}",
            f"rx {udp['packets']}  tx {udp['tx_packets']}  cliente {udp['last_client'] or '-'}",
            f"ctrl {control['mode']} {control['source']}  {control['steering']:.2f}/{control['throttle']:.2f}  auto {autonomy.get('action', '-')}",
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
        if path in {"/mode", "/drive-mode"}:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(min(length, 8192)) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_json({"ok": False, "error": "invalid json"}, status=400)
                return
            mode = payload.get("mode")
            if mode is None:
                self.send_json({"ok": False, "error": "missing mode"}, status=400)
                return
            self.send_json({"ok": True, **self.state.set_drive_mode(str(mode))})
            return
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
  <title>TP2 · Coche 4G</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Anton&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: dark;
      --bg-0: #0a0907;
      --bg-1: #11100d;
      --bg-2: #16140f;
      --line: #2b261c;
      --line-soft: #1d1a14;
      --ink: #f4ede0;
      --ink-2: #c8bfa8;
      --muted: #7d7460;
      --amber: #f5b942;
      --amber-soft: rgba(245,185,66,0.18);
      --teal: #5cd6c2;
      --red: #ff6452;
      --green: #66d28a;
      --shadow: 0 32px 60px rgba(0,0,0,0.45);
      --display: "Anton", "Bebas Neue", Impact, sans-serif;
      --body: "Manrope", -apple-system, BlinkMacSystemFont, sans-serif;
      --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    }

    * { box-sizing: border-box; }

    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      background:
        radial-gradient(1100px 600px at 80% -10%, rgba(245,185,66,0.06), transparent 60%),
        radial-gradient(900px 500px at -10% 110%, rgba(92,214,194,0.04), transparent 60%),
        var(--bg-0);
      color: var(--ink);
      font-family: var(--body);
      font-size: 13.5px;
      font-weight: 500;
      letter-spacing: 0.005em;
      overflow: hidden;
    }

    .app {
      height: 100%;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 18px;
      padding: 20px 22px 22px;
    }

    /* HEADER ----------------------------------------------------------- */
    header {
      display: grid;
      grid-template-columns: minmax(240px, auto) 1fr auto;
      align-items: center;
      gap: 22px;
      padding-bottom: 14px;
      border-bottom: 1px solid var(--line);
    }

    .brand { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
    .brand h1 {
      margin: 0;
      font-family: var(--display);
      font-weight: 400;
      font-size: 38px;
      line-height: 1;
      letter-spacing: 0.04em;
      color: var(--ink);
    }
    .brand h1 .accent { color: var(--amber); margin: 0 6px; }
    .brand h1 .sub {
      font-size: 0.58em;
      letter-spacing: 0.18em;
      color: var(--ink-2);
    }
    .brand .meta {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }

    .pills {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: center;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 9px;
      height: 30px;
      padding: 0 13px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(22,20,15,0.65);
      color: var(--ink-2);
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      font-weight: 700;
      white-space: nowrap;
    }
    .pill .label { color: var(--muted); }
    .pill .val { font-family: var(--mono); font-weight: 700; color: var(--ink); }
    .pill .dot {
      width: 7px; height: 7px; border-radius: 99px;
      background: var(--muted);
      box-shadow: 0 0 14px currentColor;
    }
    .pill.ok   { color: var(--green); } .pill.ok   .dot, .pill.ok .val   { color: var(--green); background: var(--green); }
    .pill.ok   .val { background: transparent; }
    .pill.warn { color: var(--amber); } .pill.warn .dot { background: var(--amber); color: var(--amber); }
    .pill.warn .val { color: var(--amber); }
    .pill.bad  { color: var(--red); }   .pill.bad  .dot { background: var(--red);   color: var(--red);   }
    .pill.bad  .val { color: var(--red); }

    .session {
      display: flex; align-items: center; gap: 22px;
      font-family: var(--mono);
      letter-spacing: 0.04em;
    }
    .session .group { display: flex; flex-direction: column; align-items: flex-end; }
    .session .label {
      color: var(--muted);
      font-size: 9.5px;
      letter-spacing: 0.22em;
      text-transform: uppercase;
    }
    .session .clock {
      font-size: 22px;
      color: var(--ink);
      font-weight: 500;
      line-height: 1.1;
    }
    .session .clock.amber { color: var(--amber); }

    /* MAIN GRID -------------------------------------------------------- */
    main {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 20px;
    }

    /* LEFT COLUMN: video + deck --------------------------------------- */
    .stage {
      min-height: 0;
      display: grid;
      grid-template-rows: 1fr auto;
      gap: 16px;
    }

    .video {
      position: relative;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #050403;
      overflow: hidden;
      box-shadow: var(--shadow);
      min-height: 0;
    }
    .video img {
      position: absolute; inset: 0;
      width: 100%; height: 100%;
      object-fit: contain;
      display: block;
    }
    .video::after {
      content: "";
      position: absolute; inset: 14px;
      border: 1px solid rgba(245,185,66,0.06);
      border-radius: 8px;
      pointer-events: none;
      background:
        linear-gradient(to right, var(--amber) 0 14px, transparent 14px) top left/14px 1px no-repeat,
        linear-gradient(to bottom, var(--amber) 0 14px, transparent 14px) top left/1px 14px no-repeat,
        linear-gradient(to left, var(--amber) 0 14px, transparent 14px) top right/14px 1px no-repeat,
        linear-gradient(to bottom, var(--amber) 0 14px, transparent 14px) top right/1px 14px no-repeat,
        linear-gradient(to right, var(--amber) 0 14px, transparent 14px) bottom left/14px 1px no-repeat,
        linear-gradient(to top, var(--amber) 0 14px, transparent 14px) bottom left/1px 14px no-repeat,
        linear-gradient(to left, var(--amber) 0 14px, transparent 14px) bottom right/14px 1px no-repeat,
        linear-gradient(to top, var(--amber) 0 14px, transparent 14px) bottom right/1px 14px no-repeat;
      opacity: 0.7;
    }

    .rec {
      position: absolute; right: 18px; top: 18px;
      display: flex; align-items: center; gap: 8px;
      padding: 6px 10px;
      background: rgba(8,7,5,0.82);
      border: 1px solid rgba(255,100,82,0.4);
      border-radius: 4px;
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.18em;
      color: var(--red);
    }
    .rec .blink {
      width: 8px; height: 8px; border-radius: 99px;
      background: var(--red);
      box-shadow: 0 0 14px var(--red);
      animation: blink 1.4s ease-in-out infinite;
    }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

    .hud {
      position: absolute; left: 18px; bottom: 18px;
      display: flex; gap: 8px; flex-wrap: wrap;
      pointer-events: none;
    }
    .hud .chip {
      background: rgba(8,7,5,0.78);
      border: 1px solid rgba(245,185,66,0.22);
      border-radius: 4px;
      padding: 6px 10px;
      font-family: var(--mono);
      font-size: 11.5px;
      letter-spacing: 0.04em;
      color: var(--ink);
      display: inline-flex;
      align-items: baseline;
      gap: 6px;
    }
    .hud .chip span {
      color: var(--amber);
      text-transform: uppercase;
      font-size: 9.5px;
      letter-spacing: 0.18em;
    }

    /* DECK: wheel + keys + throttle + actions */
    .deck {
      display: grid;
      grid-template-columns: auto 1fr auto;
      grid-template-rows: 1fr auto;
      column-gap: 22px;
      row-gap: 12px;
      align-items: center;
      padding: 16px 18px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(34,30,22,0.5), rgba(17,16,13,0.55));
      border-radius: 14px;
    }
    .deck .group { display: flex; flex-direction: column; gap: 10px; }
    .deck h3 {
      margin: 0;
      font-family: var(--mono);
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.22em;
      text-transform: uppercase;
      font-weight: 500;
    }

    .wheel-wrap { display: flex; align-items: center; gap: 16px; }
    .wheel {
      width: 104px; height: 104px;
      filter: drop-shadow(0 6px 14px rgba(245,185,66,0.12));
    }
    .wheel svg {
      width: 100%; height: 100%;
      transition: transform 90ms linear;
    }
    .axis-data { font-family: var(--mono); display: flex; flex-direction: column; gap: 2px; }
    .axis-data .v {
      font-size: 26px;
      color: var(--amber);
      font-weight: 700;
      line-height: 1;
    }
    .axis-data .l {
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.18em;
      text-transform: uppercase;
    }
    .axis-data .l.dir { color: var(--ink-2); letter-spacing: 0.06em; text-transform: none; font-size: 12px; }

    .keys-wrap {
      display: flex; flex-direction: column; align-items: center; gap: 8px;
      align-self: center; justify-self: center;
    }
    .keys {
      display: grid;
      grid-template-columns: repeat(3, 38px);
      grid-template-rows: 38px 38px;
      gap: 4px;
    }
    .key {
      border: 1px solid var(--line);
      background: var(--bg-2);
      border-radius: 6px;
      display: grid; place-items: center;
      font-family: var(--mono);
      font-weight: 700;
      font-size: 12px;
      color: var(--ink-2);
      letter-spacing: 0;
      transition: all 80ms ease;
    }
    .key.empty { border-color: transparent; background: transparent; }
    .key.k-w { grid-column: 2; grid-row: 1; }
    .key.k-a { grid-column: 1; grid-row: 2; }
    .key.k-s { grid-column: 2; grid-row: 2; }
    .key.k-d { grid-column: 3; grid-row: 2; }
    .key.active {
      background: var(--amber);
      color: #1c1408;
      border-color: var(--amber);
      box-shadow: 0 0 18px rgba(245,185,66,0.45);
      transform: translateY(1px);
    }
    .key.brake {
      background: linear-gradient(180deg, #4a1812, #2a0e0a);
      color: var(--red);
      border-color: rgba(255,100,82,0.45);
      box-shadow: 0 0 18px rgba(255,100,82,0.35);
    }
    .keys-caption {
      font-family: var(--mono);
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .keys-caption .kbd {
      display: inline-block;
      padding: 1px 6px;
      border: 1px solid var(--line);
      border-radius: 3px;
      color: var(--ink-2);
      margin: 0 1px;
    }

    .throttle-wrap { display: flex; align-items: center; gap: 16px; flex-direction: row-reverse; }
    .throttle-meter {
      width: 30px; height: 110px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255,100,82,0.06), transparent 50%, rgba(102,210,138,0.06));
      position: relative;
      overflow: hidden;
    }
    .throttle-meter .mid {
      position: absolute; left: -2px; right: -2px; top: 50%;
      height: 1px;
      background: var(--line);
      box-shadow: 0 0 0 0.5px rgba(245,185,66,0.18);
    }
    .throttle-meter .tick {
      position: absolute; left: 0; right: 0;
      height: 1px;
      background: rgba(245,185,66,0.12);
    }
    .throttle-meter .fill-fwd {
      position: absolute; left: 4px; right: 4px; bottom: 50%;
      height: 0%;
      background: linear-gradient(0deg, var(--green), #aef0c2);
      border-radius: 4px 4px 0 0;
      transition: height 90ms ease;
    }
    .throttle-meter .fill-rev {
      position: absolute; left: 4px; right: 4px; top: 50%;
      height: 0%;
      background: linear-gradient(180deg, var(--red), #ffb1a4);
      border-radius: 0 0 4px 4px;
      transition: height 90ms ease;
    }
    .throttle-wrap .axis-data { align-items: flex-start; }

    .deck-actions {
      grid-column: 1 / -1;
      display: flex; gap: 12px; align-items: center;
      justify-content: space-between;
      padding-top: 12px;
      border-top: 1px solid var(--line-soft);
    }
    .mode-toggle {
      display: inline-grid;
      grid-template-columns: 1fr 1fr;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 4px;
      background: var(--bg-1);
      gap: 4px;
    }
    .mode-toggle button {
      height: 36px; min-width: 120px; padding: 0 18px;
      border: 0; border-radius: 5px;
      background: transparent;
      color: var(--ink-2);
      cursor: pointer;
      font-family: var(--body);
      font-weight: 700;
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      transition: all 100ms ease;
    }
    .mode-toggle button:hover { color: var(--ink); }
    .mode-toggle button.active {
      background: var(--amber);
      color: #1c1408;
      box-shadow: 0 0 24px rgba(245,185,66,0.25);
    }

    button.stop {
      height: 44px; padding: 0 26px;
      border: 1px solid rgba(255,100,82,0.5);
      background: linear-gradient(180deg, #4a1812, #2a0e0a);
      color: var(--red);
      font-family: var(--display);
      font-size: 18px;
      letter-spacing: 0.22em;
      border-radius: 8px;
      cursor: pointer;
      text-transform: uppercase;
      transition: all 120ms ease;
    }
    button.stop:hover {
      background: linear-gradient(180deg, #6b2018, #3a1310);
      box-shadow: 0 0 24px rgba(255,100,82,0.25);
    }
    button.stop:active { transform: translateY(1px); }

    /* RIGHT COLUMN: telemetry stack ----------------------------------- */
    .side {
      min-height: 0;
      overflow-y: auto;
      display: grid;
      align-content: start;
      gap: 14px;
      padding-right: 4px;
      scrollbar-width: thin;
      scrollbar-color: var(--line) transparent;
    }
    .side::-webkit-scrollbar { width: 6px; }
    .side::-webkit-scrollbar-track { background: transparent; }
    .side::-webkit-scrollbar-thumb { background: var(--line); border-radius: 99px; }

    .card {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(22,20,15,0.7), rgba(17,16,13,0.7));
      border-radius: 12px;
      padding: 16px 16px 14px;
    }
    .card h2 {
      margin: 0 0 12px 0;
      font-family: var(--mono);
      font-weight: 500;
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.28em;
      text-transform: uppercase;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .card h2 .tag {
      color: var(--amber);
      font-family: var(--mono);
      font-weight: 700;
      font-size: 10px;
      letter-spacing: 0.18em;
    }
    .card h3 {
      margin: 14px 0 8px;
      font-family: var(--mono);
      font-weight: 500;
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.24em;
      text-transform: uppercase;
    }

    .row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: baseline;
      padding: 6px 0;
      border-bottom: 1px dashed var(--line-soft);
    }
    .row:last-child { border-bottom: 0; }
    .row .k {
      color: var(--muted);
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      font-size: 11px;
    }
    .row .v {
      font-family: var(--mono);
      color: var(--ink);
      font-weight: 500;
      text-align: right;
      font-size: 12.5px;
      font-variant-numeric: tabular-nums;
    }
    .row .v.amber { color: var(--amber); }
    .row .v.green { color: var(--green); }
    .row .v.red { color: var(--red); }
    .row .v.muted { color: var(--muted); }

    /* battery */
    .battery .gauge {
      height: 10px;
      border-radius: 4px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: var(--bg-2);
      position: relative;
      margin-top: 4px;
    }
    .battery .gauge .fill {
      height: 100%; width: 0%;
      background: linear-gradient(90deg, var(--green), #aef0c2);
      transition: width 240ms ease, background 240ms ease;
    }

    /* sparkline */
    .spark-wrap {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: end;
      gap: 14px;
      padding: 8px 0 4px;
    }
    .spark {
      height: 38px;
      width: 100%;
    }
    .spark path { fill: none; stroke: var(--amber); stroke-width: 1.4; stroke-linejoin: round; stroke-linecap: round; }
    .spark .area { fill: rgba(245,185,66,0.13); stroke: none; }
    .spark .grid { stroke: var(--line-soft); stroke-width: 1; stroke-dasharray: 2 4; }
    .spark.teal path { stroke: var(--teal); }
    .spark.teal .area { fill: rgba(92,214,194,0.12); }

    .spark-data { font-family: var(--mono); text-align: right; }
    .spark-data .v {
      font-size: 20px;
      color: var(--ink);
      font-weight: 700;
      line-height: 1;
    }
    .spark-data .v small {
      font-size: 0.55em;
      color: var(--muted);
      font-weight: 500;
      margin-left: 3px;
    }
    .spark-data .l {
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.16em;
      text-transform: uppercase;
      margin-top: 4px;
    }

    /* detection list */
    .detections { display: grid; gap: 6px; max-height: 200px; overflow: auto; padding-right: 2px; }
    .detections::-webkit-scrollbar { width: 4px; }
    .detections::-webkit-scrollbar-thumb { background: var(--line); border-radius: 99px; }
    .det {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 12px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(8,7,5,0.4);
    }
    .det .name {
      font-weight: 700;
      letter-spacing: 0.02em;
      font-size: 12.5px;
      color: var(--ink);
    }
    .det .conf {
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: var(--mono);
      font-size: 12px;
      color: var(--ink);
    }
    .det .conf .meter {
      width: 56px; height: 4px;
      background: var(--line);
      border-radius: 2px;
      overflow: hidden;
    }
    .det .conf .meter .fill {
      height: 100%;
      background: var(--amber);
      width: 0%;
      transition: width 240ms ease;
    }
    .det.empty {
      text-align: center;
      color: var(--muted);
      font-style: italic;
      font-size: 12px;
      border-style: dashed;
      grid-template-columns: 1fr;
    }

    /* responsive */
    @media (max-width: 1080px) {
      html, body { overflow: auto; }
      .app { height: auto; min-height: 100%; }
      header { grid-template-columns: 1fr; gap: 14px; }
      .pills { justify-content: flex-start; }
      .session { justify-content: flex-start; gap: 18px; }
      .session .group { align-items: flex-start; }
      main { grid-template-columns: 1fr; }
      .video { aspect-ratio: 16 / 9; }
      .side { max-height: none; overflow: visible; }
    }
    @media (max-width: 720px) {
      .deck { grid-template-columns: 1fr; row-gap: 18px; }
      .deck .group { align-items: center; }
      .deck-actions { flex-direction: column; align-items: stretch; }
      .mode-toggle button { min-width: 0; }
      .brand h1 { font-size: 30px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <h1>TP2<span class="accent">·</span><span class="sub">COCHE 4G</span></h1>
        <span class="meta">EPC · Roboflow · UDP 20001</span>
      </div>

      <div class="pills">
        <div class="pill warn" id="pill-link"><span class="dot"></span><span class="label">4G</span><span class="val" id="pill-link-val">--</span></div>
        <div class="pill warn" id="pill-video"><span class="dot"></span><span class="label">Vídeo</span><span class="val" id="pill-video-val">--</span></div>
        <div class="pill warn" id="pill-ai"><span class="dot"></span><span class="label">IA</span><span class="val" id="pill-ai-val">--</span></div>
        <div class="pill bad" id="pill-control"><span class="dot"></span><span class="label">Control</span><span class="val" id="pill-control-val">OFF</span></div>
      </div>

      <div class="session">
        <div class="group">
          <span class="label">Sesión</span>
          <span class="clock amber" id="session-clock">00:00:00</span>
        </div>
        <div class="group">
          <span class="label">Hora</span>
          <span class="clock" id="wall-clock">--:--:--</span>
        </div>
      </div>
    </header>

    <main>
      <section class="stage">
        <div class="video">
          <img id="video" src="/video.mjpg" alt="Cámara del coche">
          <div class="rec"><span class="blink"></span><span>EN VIVO</span></div>
          <div class="hud">
            <div class="chip"><span>FPS</span><strong id="hud-fps">--</strong></div>
            <div class="chip"><span>Lat</span><strong id="hud-lat">-- ms</strong></div>
            <div class="chip"><span>Det</span><strong id="hud-det">--</strong></div>
            <div class="chip"><span>Frame</span><strong id="hud-frame">--</strong></div>
          </div>
        </div>

        <div class="deck">
          <div class="group wheel-wrap">
            <div class="wheel">
              <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" id="wheel-svg" aria-hidden="true">
                <defs>
                  <linearGradient id="rim" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="#3a342a"/>
                    <stop offset="100%" stop-color="#1a1610"/>
                  </linearGradient>
                </defs>
                <circle cx="50" cy="50" r="45" fill="url(#rim)" stroke="#2b261c" stroke-width="2"/>
                <circle cx="50" cy="50" r="36" fill="none" stroke="#0e0c08" stroke-width="3"/>
                <g stroke="#f5b942" stroke-width="3" stroke-linecap="round">
                  <line x1="50" y1="50" x2="50" y2="14"/>
                  <line x1="50" y1="50" x2="20" y2="68"/>
                  <line x1="50" y1="50" x2="80" y2="68"/>
                </g>
                <circle cx="50" cy="50" r="9" fill="#0e0c08" stroke="#f5b942" stroke-width="2"/>
                <circle cx="50" cy="14" r="2.5" fill="#f5b942"/>
              </svg>
            </div>
            <div class="axis-data">
              <span class="l">Giro</span>
              <span class="v" id="steer-val">0.25</span>
              <span class="l dir" id="steer-dir">centrado</span>
            </div>
          </div>

          <div class="group keys-wrap">
            <h3>Teclas</h3>
            <div class="keys">
              <div class="key empty"></div>
              <div class="key k-w" data-key="w">W</div>
              <div class="key empty"></div>
              <div class="key k-a" data-key="a">A</div>
              <div class="key k-s" data-key="s">S</div>
              <div class="key k-d" data-key="d">D</div>
            </div>
            <div class="keys-caption">freno <span class="kbd">␣</span><span class="kbd">X</span></div>
          </div>

          <div class="group throttle-wrap">
            <div class="throttle-meter" aria-label="Acelerador">
              <div class="tick" style="top:25%"></div>
              <div class="mid"></div>
              <div class="tick" style="top:75%"></div>
              <div class="fill-fwd" id="thr-fwd"></div>
              <div class="fill-rev" id="thr-rev"></div>
            </div>
            <div class="axis-data">
              <span class="l">Acelerador</span>
              <span class="v" id="thr-val">0.00</span>
              <span class="l dir" id="thr-dir">parado</span>
            </div>
          </div>

          <div class="deck-actions">
            <div class="mode-toggle" role="group" aria-label="Modo de conducción">
              <button type="button" id="mode-manual" class="active">Manual</button>
              <button type="button" id="mode-auto">Autónomo</button>
            </div>
            <button type="button" class="stop" id="stop">Stop</button>
          </div>
        </div>
      </section>

      <aside class="side">
        <section class="card">
          <h2>Inferencia <span class="tag" id="ai-tag">--</span></h2>
          <div class="row"><span class="k">Backend</span><span class="v muted" id="ai-backend">--</span></div>
          <div class="row"><span class="k">Modelo</span><span class="v muted" id="ai-model">--</span></div>
          <div class="row"><span class="k">Estado</span><span class="v" id="ai-status">--</span></div>

          <div class="spark-wrap">
            <svg class="spark" id="spark-lat" viewBox="0 0 200 38" preserveAspectRatio="none">
              <line class="grid" x1="0" y1="19" x2="200" y2="19"/>
              <path class="area" d=""/>
              <path d=""/>
            </svg>
            <div class="spark-data">
              <div class="v"><span id="ai-latency">--</span><small>ms</small></div>
              <div class="l">Latencia IA</div>
            </div>
          </div>
          <div class="spark-wrap">
            <svg class="spark teal" id="spark-fps" viewBox="0 0 200 38" preserveAspectRatio="none">
              <line class="grid" x1="0" y1="19" x2="200" y2="19"/>
              <path class="area" d=""/>
              <path d=""/>
            </svg>
            <div class="spark-data">
              <div class="v"><span id="ai-fps">--</span><small>fps</small></div>
              <div class="l">Vídeo</div>
            </div>
          </div>

          <h3>Detecciones</h3>
          <div class="detections" id="detections">
            <div class="det empty"><span>Esperando inferencia…</span></div>
          </div>
        </section>

        <section class="card">
          <h2>Autonomía</h2>
          <div class="row"><span class="k">Modo</span><span class="v amber" id="auto-mode">--</span></div>
          <div class="row"><span class="k">Acción</span><span class="v" id="auto-action">--</span></div>
          <div class="row"><span class="k">Señal</span><span class="v" id="auto-target">--</span></div>
          <div class="row"><span class="k">Zona / Distancia</span><span class="v muted" id="auto-zone">--</span></div>
          <div class="row"><span class="k">Motivo</span><span class="v muted" id="auto-reason">--</span></div>
        </section>

        <section class="card">
          <h2>Enlace 4G</h2>
          <div class="row"><span class="k">UDP</span><span class="v muted" id="link-bind">--</span></div>
          <div class="row"><span class="k">Cliente</span><span class="v" id="link-client">--</span></div>
          <div class="row"><span class="k">Último paquete</span><span class="v" id="link-last">--</span></div>
          <div class="row"><span class="k">RX</span><span class="v" id="link-rx">--</span></div>
          <div class="row"><span class="k">TX</span><span class="v" id="link-tx">--</span></div>
          <div class="row"><span class="k">Errores</span><span class="v" id="link-err">0</span></div>
        </section>

        <section class="card">
          <h2>Coche</h2>
          <div class="battery">
            <div class="row" style="border:0; padding-bottom: 4px;">
              <span class="k">Batería</span><span class="v amber" id="bat-val">--</span>
            </div>
            <div class="gauge"><div class="fill" id="bat-fill"></div></div>
          </div>
          <div class="row" style="margin-top: 12px;"><span class="k">Origen control</span><span class="v" id="ctrl-source">--</span></div>
          <div class="row"><span class="k">Watchdog</span><span class="v" id="ctrl-watch">--</span></div>
          <div class="row"><span class="k">Stream activo</span><span class="v" id="stream-clients">--</span></div>
          <div class="row"><span class="k">Posts control</span><span class="v" id="control-posts">--</span></div>
        </section>
      </aside>
    </main>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const els = {
      pillLink: $('pill-link'),    pillLinkVal: $('pill-link-val'),
      pillVideo: $('pill-video'),  pillVideoVal: $('pill-video-val'),
      pillAi: $('pill-ai'),        pillAiVal: $('pill-ai-val'),
      pillCtrl: $('pill-control'), pillCtrlVal: $('pill-control-val'),
      sessionClock: $('session-clock'),
      wallClock: $('wall-clock'),

      hudFps: $('hud-fps'), hudLat: $('hud-lat'), hudDet: $('hud-det'), hudFrame: $('hud-frame'),

      wheelSvg: $('wheel-svg'),
      steerVal: $('steer-val'), steerDir: $('steer-dir'),
      thrFwd: $('thr-fwd'), thrRev: $('thr-rev'),
      thrVal: $('thr-val'), thrDir: $('thr-dir'),

      modeManual: $('mode-manual'), modeAuto: $('mode-auto'), stop: $('stop'),

      aiTag: $('ai-tag'),
      aiBackend: $('ai-backend'), aiModel: $('ai-model'), aiStatus: $('ai-status'),
      aiLatency: $('ai-latency'), aiFps: $('ai-fps'),
      sparkLat: $('spark-lat'), sparkFps: $('spark-fps'),
      detections: $('detections'),

      autoMode: $('auto-mode'), autoAction: $('auto-action'),
      autoTarget: $('auto-target'), autoZone: $('auto-zone'), autoReason: $('auto-reason'),

      linkBind: $('link-bind'), linkClient: $('link-client'), linkLast: $('link-last'),
      linkRx: $('link-rx'), linkTx: $('link-tx'), linkErr: $('link-err'),

      batVal: $('bat-val'), batFill: $('bat-fill'),
      ctrlSource: $('ctrl-source'), ctrlWatch: $('ctrl-watch'),
      streamClients: $('stream-clients'), controlPosts: $('control-posts'),
    };

    /* state */
    const NEUTRAL_STEERING = 0.25;
    const KEY_CODES = ['w','a','s','d','x',' ','arrowup','arrowdown','arrowleft','arrowright'];
    const keys = new Set();
    let driveMode = 'manual';
    let lastSent = { steering: NEUTRAL_STEERING, throttle: 0.0 };
    let viewControl = { steering: NEUTRAL_STEERING, throttle: 0.0 };

    /* sparkline buffers */
    const LAT_BUF = [], FPS_BUF = [];
    const BUF_LEN = 60;
    let lastFrames = null, lastFramesAt = null, fpsEma = 0;

    /* ---------- utilities ---------- */
    function setPillState(el, state) {
      el.classList.remove('ok','warn','bad');
      el.classList.add(state);
    }
    function fmtTime(seconds) {
      seconds = Math.max(0, Math.floor(seconds));
      const h = Math.floor(seconds / 3600);
      const m = Math.floor((seconds % 3600) / 60);
      const s = seconds % 60;
      return [h, m, s].map(n => String(n).padStart(2,'0')).join(':');
    }
    function clampNum(v, a, b) { return Math.max(a, Math.min(b, v)); }
    function nfmt(v, d=2) { return v == null || Number.isNaN(+v) ? '--' : Number(v).toFixed(d); }

    function tickWallClock() {
      const d = new Date();
      els.wallClock.textContent = [d.getHours(), d.getMinutes(), d.getSeconds()]
        .map(n => String(n).padStart(2,'0')).join(':');
    }
    setInterval(tickWallClock, 1000); tickWallClock();

    /* ---------- mode + control ---------- */
    function renderMode(mode) {
      driveMode = mode === 'autonomous' ? 'autonomous' : 'manual';
      els.modeManual.classList.toggle('active', driveMode === 'manual');
      els.modeAuto.classList.toggle('active', driveMode === 'autonomous');
    }

    function axisFromKeys() {
      let throttle = 0.0;
      if (keys.has('w') || keys.has('arrowup')) throttle = 0.6;
      if (keys.has('s') || keys.has('arrowdown')) throttle = -0.5;
      if (keys.has('x') || keys.has(' ')) throttle = -0.9;
      let steering = NEUTRAL_STEERING;
      const left = keys.has('a') || keys.has('arrowleft');
      const right = keys.has('d') || keys.has('arrowright');
      if (left && !right) steering = 1.0;
      if (right && !left) steering = -1.0;
      return { steering, throttle };
    }

    function renderControl(steering, throttle) {
      const offset = clampNum(steering - NEUTRAL_STEERING, -1, 1);
      const rot = -offset * 130;
      els.wheelSvg.style.transform = `rotate(${rot.toFixed(1)}deg)`;
      els.steerVal.textContent = nfmt(steering, 2);
      els.steerDir.textContent =
        offset > 0.05  ? 'izquierda · ' + Math.round(offset * 100) + '%'
      : offset < -0.05 ? 'derecha · '   + Math.round(-offset * 100) + '%'
      :                  'centrado';

      const fwd = clampNum(Math.max(0, throttle), 0, 1);
      const rev = clampNum(Math.max(0, -throttle), 0, 1);
      els.thrFwd.style.height = (fwd * 50).toFixed(1) + '%';
      els.thrRev.style.height = (rev * 50).toFixed(1) + '%';
      els.thrVal.textContent = nfmt(throttle, 2);
      els.thrDir.textContent =
        throttle >  0.05 ? 'avanza · ' + Math.round(throttle * 100) + '%'
      : throttle < -0.05 ? (throttle < -0.7 ? 'freno fuerte' : 'retroceder · ' + Math.round(-throttle * 100) + '%')
      :                    'parado';
    }

    /* highlight WASD on keypress */
    function paintKeys() {
      document.querySelectorAll('.key[data-key]').forEach(k => {
        const isBrake = (k.dataset.key === 's') && (keys.has(' ') || keys.has('x'));
        const pressed = keys.has(k.dataset.key) || (k.dataset.key === 'w' && keys.has('arrowup'))
          || (k.dataset.key === 'a' && keys.has('arrowleft'))
          || (k.dataset.key === 's' && keys.has('arrowdown'))
          || (k.dataset.key === 'd' && keys.has('arrowright'));
        k.classList.toggle('active', pressed && !isBrake);
        k.classList.toggle('brake', isBrake);
      });
    }

    async function postControl(control) {
      if (driveMode !== 'manual') return;
      try {
        const res = await fetch('/control', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(control),
          cache: 'no-store',
        });
        if (!res.ok) throw new Error('http ' + res.status);
      } catch (_) {
        setPillState(els.pillCtrl, 'bad');
      }
    }

    async function postMode(mode) {
      renderMode(mode);
      keys.clear(); paintKeys();
      try {
        const res = await fetch('/mode', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({mode}),
          cache: 'no-store',
        });
        if (!res.ok) throw new Error('http ' + res.status);
      } catch (_) {
        setPillState(els.pillCtrl, 'bad');
      }
    }

    async function neutral() {
      keys.clear(); paintKeys();
      renderMode('manual');
      lastSent = { steering: NEUTRAL_STEERING, throttle: 0.0 };
      renderControl(NEUTRAL_STEERING, 0.0);
      setPillState(els.pillCtrl, 'bad');
      els.pillCtrlVal.textContent = 'OFF';
      try { await fetch('/control/neutral', { method: 'POST', cache: 'no-store' }); } catch (_) {}
    }

    window.addEventListener('keydown', (event) => {
      const key = event.key.toLowerCase();
      if (KEY_CODES.includes(key)) {
        event.preventDefault();
        keys.add(key);
        paintKeys();
      }
    });
    window.addEventListener('keyup', (event) => {
      keys.delete(event.key.toLowerCase());
      paintKeys();
    });
    window.addEventListener('blur', neutral);
    document.addEventListener('visibilitychange', () => { if (document.hidden) neutral(); });

    els.modeManual.addEventListener('click', () => postMode('manual'));
    els.modeAuto.addEventListener('click', () => postMode('autonomous'));
    els.stop.addEventListener('click', neutral);

    /* manual control loop */
    setInterval(() => {
      if (driveMode === 'manual') {
        const c = axisFromKeys();
        lastSent = c;
        renderControl(c.steering, c.throttle);
        postControl(c);
      }
    }, 50);

    /* ---------- sparklines ---------- */
    function pushBuf(buf, val) {
      buf.push(val);
      if (buf.length > BUF_LEN) buf.shift();
    }
    function drawSpark(svg, values, opts) {
      const opts2 = opts || {};
      const W = 200, H = 38;
      const paths = svg.querySelectorAll('path');
      const area = paths[0], line = paths[1];
      if (!values.length) {
        area.setAttribute('d',''); line.setAttribute('d',''); return;
      }
      const fixedMax = opts2.fixedMax ? opts2.fixedMax : Math.max(...values, opts2.minMax || 1);
      const max = fixedMax * 1.1;
      const n = values.length;
      const pts = values.map((v, i) => {
        const x = n > 1 ? (i / (n - 1)) * W : W;
        const y = H - clampNum(v / max, 0, 1) * (H - 4) - 2;
        return [x, y];
      });
      const d = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ');
      const a = d + ' L' + W + ',' + H + ' L0,' + H + ' Z';
      line.setAttribute('d', d);
      area.setAttribute('d', a);
    }

    /* ---------- status polling ---------- */
    async function pollStatus() {
      try {
        const res = await fetch('/status.json', { cache: 'no-store' });
        const data = await res.json();
        const now = performance.now() / 1000;

        /* link */
        const pktAge = data.udp.last_packet_age_sec;
        const linkOk = pktAge !== null && pktAge < 1.2;
        const linkWarn = pktAge !== null && pktAge < 3.0;
        setPillState(els.pillLink, linkOk ? 'ok' : (linkWarn ? 'warn' : 'bad'));
        els.pillLinkVal.textContent = linkOk ? 'ONLINE' : (linkWarn ? 'LENTO' : 'SIN RX');

        /* video + fps */
        const vid = data.video;
        const videoAge = vid.age_sec;
        const videoOk = vid.has_video && (videoAge === null || videoAge < 1.5);
        setPillState(els.pillVideo, videoOk ? 'ok' : 'warn');
        els.pillVideoVal.textContent = videoOk ? vid.frames : 'SIN';

        if (lastFrames !== null && lastFramesAt !== null) {
          const dt = now - lastFramesAt;
          if (dt > 0.2) {
            const inst = (vid.frames - lastFrames) / dt;
            fpsEma = fpsEma === 0 ? inst : (0.65 * fpsEma + 0.35 * inst);
            pushBuf(FPS_BUF, Math.max(0, fpsEma));
            lastFrames = vid.frames; lastFramesAt = now;
          }
        } else {
          lastFrames = vid.frames; lastFramesAt = now;
        }
        const fpsShown = videoOk ? fpsEma : 0;
        els.aiFps.textContent = fpsShown ? fpsShown.toFixed(1) : '--';
        els.hudFps.textContent = fpsShown ? fpsShown.toFixed(0) : '--';
        drawSpark(els.sparkFps, FPS_BUF, { minMax: 30 });

        /* inference */
        const inf = data.inference;
        const aiOk = ['ready','running','waiting-frame'].includes(inf.status);
        setPillState(els.pillAi, aiOk ? 'ok' : (inf.status === 'starting' ? 'warn' : 'bad'));
        const statusEs = {
          'ready': 'lista', 'running': 'analizando', 'waiting-frame': 'esperando',
          'starting': 'iniciando', 'offline': 'offline',
          'disabled': 'deshabilitada', 'error': 'error',
        }[inf.status] || inf.status;
        els.pillAiVal.textContent = statusEs;
        els.aiTag.textContent = inf.detections + ' obj';
        els.aiStatus.textContent = inf.error || statusEs;
        els.aiBackend.textContent = (inf.backend && inf.backend.api_url) || '--';
        els.aiModel.textContent = (inf.backend && inf.backend.model_id) || '--';
        if (inf.latency_ms != null) {
          pushBuf(LAT_BUF, inf.latency_ms);
          els.aiLatency.textContent = inf.latency_ms;
          els.hudLat.textContent = inf.latency_ms + ' ms';
        } else {
          els.aiLatency.textContent = '--';
          els.hudLat.textContent = '-- ms';
        }
        drawSpark(els.sparkLat, LAT_BUF, { minMax: 200 });

        els.hudDet.textContent = inf.detections;
        els.hudFrame.textContent = vid.frames;

        /* detections list */
        els.detections.innerHTML = '';
        const preds = inf.predictions || [];
        if (!preds.length) {
          const empty = document.createElement('div');
          empty.className = 'det empty';
          empty.innerHTML = '<span>Sin detecciones</span>';
          els.detections.appendChild(empty);
        } else {
          for (const p of preds.slice(0, 8)) {
            const conf = p.confidence === undefined ? 0 : Number(p.confidence);
            const row = document.createElement('div');
            row.className = 'det';
            row.innerHTML =
              '<span class="name">' + (p.class || 'objeto') + '</span>' +
              '<span class="conf">' +
                '<span class="meter"><span class="fill" style="width:' + (conf * 100).toFixed(0) + '%"></span></span>' +
                '<span>' + (conf * 100).toFixed(0) + '%</span>' +
              '</span>';
            els.detections.appendChild(row);
          }
        }

        /* control + autonomy pills + values */
        renderMode(data.control.mode || 'manual');
        const autoDecision = (data.autonomy && data.autonomy.decision) || {};
        const autoActive = data.control.mode === 'autonomous' && autoDecision.active;
        if (data.control.mode === 'autonomous') {
          setPillState(els.pillCtrl, autoActive ? 'ok' : 'warn');
          els.pillCtrlVal.textContent = autoActive ? 'AUTO' : 'SAFE';
        } else {
          setPillState(els.pillCtrl, data.control.armed ? 'ok' : 'bad');
          els.pillCtrlVal.textContent = data.control.armed ? 'ON' : 'OFF';
        }

        const remoteSteer = Number(data.control.steering);
        const remoteThr = Number(data.control.throttle);
        if (driveMode === 'manual' && data.control.armed) {
          renderControl(lastSent.steering, lastSent.throttle);
        } else {
          renderControl(remoteSteer, remoteThr);
        }

        /* autonomy card */
        els.autoMode.textContent = data.control.mode === 'autonomous' ? 'autónomo' : 'manual';
        const actionEs = {
          'continue': 'avanzar',
          'turn-left': 'girar izquierda', 'turn-right': 'girar derecha',
          'prepare-left': 'preparar izquierda', 'prepare-right': 'preparar derecha',
          'safe-neutral': 'neutro seguro', 'crawl': 'avance lento',
          'slow': 'lento', 'cruise': 'crucero', 'stop': 'detenido',
          'brake': 'frenar',
        }[autoDecision.action] || (autoDecision.action || '--');
        els.autoAction.textContent = actionEs;
        const tgt = autoDecision.target;
        els.autoTarget.textContent = tgt ? (tgt.class + ' · ' + (Number(tgt.confidence)*100).toFixed(0) + '%') : '--';
        els.autoZone.textContent = tgt ? (tgt.zone + ' · ' + tgt.distance) : '--';
        els.autoReason.textContent = autoDecision.reason || '--';

        /* link card */
        els.linkBind.textContent = data.udp.bind;
        els.linkClient.textContent = data.udp.last_client || '--';
        const types = data.udp.last_packet_type || '--';
        const ageS = data.udp.last_packet_age_sec;
        els.linkLast.textContent = ageS == null ? types : (types + ' · ' + Number(ageS).toFixed(2) + ' s');
        const pkts = data.udp.packets || {};
        const totalRx = Object.values(pkts).reduce((a, b) => a + (Number(b) || 0), 0);
        const breakdown = Object.entries(pkts).map(([k, v]) => k + ':' + v).join(' ');
        els.linkRx.textContent = totalRx + (breakdown ? ' · ' + breakdown : '');
        els.linkTx.textContent = data.udp.tx_packets;
        els.linkErr.textContent = (data.udp.bad_packets || 0) + (vid.decode_errors ? ' · ' + vid.decode_errors + ' dec' : '');

        /* car card */
        const bat = data.car && data.car.battery;
        if (bat == null) {
          els.batVal.textContent = '--';
          els.batFill.style.width = '0%';
        } else {
          const b = Number(bat);
          els.batVal.textContent = b.toFixed(2);
          const pct = b > 1 ? clampNum(b, 0, 100) : clampNum(b * 100, 0, 100);
          els.batFill.style.width = pct.toFixed(0) + '%';
          els.batFill.style.background = pct < 20
            ? 'linear-gradient(90deg, #ff6452, #ffb1a4)'
            : pct < 45
            ? 'linear-gradient(90deg, #f5b942, #ffd98a)'
            : 'linear-gradient(90deg, #66d28a, #aef0c2)';
        }

        els.ctrlSource.textContent =
          data.control.source + ' · ' + nfmt(remoteSteer, 2) + ' / ' + nfmt(remoteThr, 2);
        els.ctrlWatch.textContent = nfmt(data.control.updated_age_sec, 2) + ' s';
        els.streamClients.textContent = (data.web && data.web.stream_clients) || 0;
        els.controlPosts.textContent = (data.web && data.web.control_posts) || 0;

        els.sessionClock.textContent = fmtTime(data.uptime_sec || 0);
      } catch (err) {
        setPillState(els.pillLink, 'bad'); els.pillLinkVal.textContent = 'ERR';
        setPillState(els.pillVideo, 'bad'); els.pillVideoVal.textContent = '--';
      }
    }

    pollStatus();
    setInterval(pollStatus, 250);
    renderControl(NEUTRAL_STEERING, 0.0);
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
