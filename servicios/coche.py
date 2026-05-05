from __future__ import annotations

import json
import math
import os
import pickle
import signal
import socket
import struct
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
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
    AutonomousController,
    AutonomousDecision,
)
from lane_detector import (  # noqa: E402
    LaneDetector,
    LaneDetectorConfig,
    LaneGuidance,
    draw_lane_overlay,
)
from lidar_processor import (  # noqa: E402
    LidarConfig,
    LidarScan,
    analyze_lidar_scan,
    lidar_status_points,
    normalize_lidar_payload,
)
from roboflow_runtime import (  # noqa: E402
    InferenceConfig,
    create_client,
    draw_predictions_on_image,
    extract_predictions,
    infer_one_frame,
    local_endpoint_reachable,
)
from session_replayer import ReplayerHandler, SessionCatalog  # noqa: E402


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


def env_csv_set(name: str, default: set[str]) -> set[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return set(default)
    values = {item.strip() for item in raw.split(",")}
    return {item for item in values if item}


BIND_IP = os.getenv("TP2_BIND_IP", "172.16.0.1")
BIND_PORT = env_int("TP2_BIND_PORT", 20001)
UDP_RECV_BYTES = env_int("TP2_UDP_RECV_BYTES", 131072)

WEB_HOST = os.getenv("TP2_WEB_HOST", "0.0.0.0")
WEB_PORT = env_int("TP2_WEB_PORT", 8088)
ENABLE_WEB_VIEW = env_bool("TP2_ENABLE_WEB_VIEW", True)
ENABLE_WEB_CONTROL = env_bool("TP2_ENABLE_WEB_CONTROL", True)
ENABLE_INFERENCE = env_bool("TP2_ENABLE_INFERENCE", True)

NEUTRAL_STEERING = env_float("TP2_NEUTRAL_STEERING", 0.25)
STEERING_TRIM = env_float("TP2_STEERING_TRIM", -0.24)
NEUTRAL_THROTTLE = env_float("TP2_NEUTRAL_THROTTLE", 0.0)
MANUAL_FORWARD_THROTTLE = env_float("TP2_MANUAL_FORWARD_THROTTLE", 0.60)
MANUAL_REVERSE_THROTTLE = env_float("TP2_MANUAL_REVERSE_THROTTLE", -0.50)
MANUAL_BRAKE_THROTTLE = env_float("TP2_MANUAL_BRAKE_THROTTLE", -0.90)
CONTROL_TIMEOUT_SEC = env_float("TP2_WEB_CONTROL_TIMEOUT_SEC", 0.45)
CONTROL_TX_HZ = max(1.0, env_float("TP2_CONTROL_TX_HZ", 20.0))
CLIENT_ADDR_TTL_SEC = env_float("TP2_CLIENT_ADDR_TTL_SEC", 3.0)

INFERENCE_MIN_INTERVAL_SEC = env_float("TP2_INFERENCE_MIN_INTERVAL_SEC", 0.07)
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
    min_area_ratio=env_float("TP2_AUTONOMOUS_MIN_AREA_RATIO", 0.003),
    near_area_ratio=env_float("TP2_AUTONOMOUS_NEAR_AREA_RATIO", 0.030),
    center_left=env_float("TP2_AUTONOMOUS_CENTER_LEFT", 0.40),
    center_right=env_float("TP2_AUTONOMOUS_CENTER_RIGHT", 0.60),
    neutral_steering=NEUTRAL_STEERING,
    neutral_throttle=NEUTRAL_THROTTLE,
    crawl_throttle=env_float("TP2_AUTONOMOUS_CRAWL_THROTTLE", 0.65),
    slow_throttle=env_float("TP2_AUTONOMOUS_SLOW_THROTTLE", 0.65),
    turn_throttle=env_float("TP2_AUTONOMOUS_TURN_THROTTLE", 0.65),
    cruise_throttle=env_float("TP2_AUTONOMOUS_CRUISE_THROTTLE", 0.65),
    fast_throttle=env_float("TP2_AUTONOMOUS_FAST_THROTTLE", 0.65),
    left_steering=env_float("TP2_AUTONOMOUS_LEFT_STEERING", 1.0),
    right_steering=env_float("TP2_AUTONOMOUS_RIGHT_STEERING", -1.0),
    confirm_frames=env_int("TP2_AUTONOMOUS_CONFIRM_FRAMES", 1),
    safety_confirm_frames=env_int("TP2_AUTONOMOUS_SAFETY_CONFIRM_FRAMES", 1),
    max_track_age_sec=env_float("TP2_AUTONOMOUS_MAX_TRACK_AGE_SEC", 1.2),
    track_memory_sec=env_float("TP2_AUTONOMOUS_TRACK_MEMORY_SEC", 0.45),
    match_iou=env_float("TP2_AUTONOMOUS_MATCH_IOU", 0.14),
    match_center_distance=env_float("TP2_AUTONOMOUS_MATCH_CENTER_DISTANCE", 0.18),
    ambiguous_score_ratio=env_float("TP2_AUTONOMOUS_AMBIGUOUS_SCORE_RATIO", 0.82),
    stop_hold_sec=env_float("TP2_AUTONOMOUS_STOP_HOLD_SEC", 1.15),
    turn_hold_sec=env_float("TP2_AUTONOMOUS_TURN_HOLD_SEC", 1.20),
    turn_pulse_enabled=env_bool("TP2_AUTONOMOUS_TURN_PULSE_ENABLED", True),
    turn_degrees=env_int("TP2_AUTONOMOUS_TURN_DEGREES", 90),
    cooldown_sec=env_float("TP2_AUTONOMOUS_COOLDOWN_SEC", 0.85),
    distance_scale=env_float("TP2_AUTONOMOUS_DISTANCE_SCALE", 0.32),
    steering_rate_per_sec=env_float("TP2_AUTONOMOUS_STEERING_RATE_PER_SEC", 2.4),
    throttle_rate_per_sec=env_float("TP2_AUTONOMOUS_THROTTLE_RATE_PER_SEC", 1.0),
    brake_rate_per_sec=env_float("TP2_AUTONOMOUS_BRAKE_RATE_PER_SEC", 3.0),
    dry_run=env_bool("TP2_AUTONOMOUS_DRY_RUN", False),
)

LANE_CONFIG = LaneDetectorConfig(
    enabled=env_bool("TP2_LANE_ASSIST_ENABLED", True),
    roi_top_ratio=env_float("TP2_LANE_ROI_TOP_RATIO", 0.34),
    roi_bottom_margin_ratio=env_float("TP2_LANE_ROI_BOTTOM_MARGIN_RATIO", 0.02),
    target_center_x=env_float("TP2_LANE_TARGET_CENTER_X", 0.50),
    lower_sample_y=env_float("TP2_LANE_LOWER_SAMPLE_Y", 0.86),
    upper_sample_y=env_float("TP2_LANE_UPPER_SAMPLE_Y", 0.58),
    hsv_lower=(
        env_int("TP2_LANE_H_MIN", 42),
        env_int("TP2_LANE_S_MIN", 45),
        env_int("TP2_LANE_V_MIN", 55),
    ),
    hsv_upper=(
        env_int("TP2_LANE_H_MAX", 105),
        env_int("TP2_LANE_S_MAX", 255),
        env_int("TP2_LANE_V_MAX", 255),
    ),
    road_gray_max=env_int("TP2_LANE_ROAD_GRAY_MAX", 125),
    road_context_dilate_px=env_int("TP2_LANE_ROAD_CONTEXT_DILATE_PX", 33),
    min_component_area_ratio=env_float("TP2_LANE_MIN_COMPONENT_AREA_RATIO", 0.00016),
    min_line_height_ratio=env_float("TP2_LANE_MIN_LINE_HEIGHT_RATIO", 0.11),
    max_fit_error_ratio=env_float("TP2_LANE_MAX_FIT_ERROR_RATIO", 0.055),
    max_curve_fit_error_ratio=env_float("TP2_LANE_MAX_CURVE_FIT_ERROR_RATIO", 0.12),
    cluster_px_ratio=env_float("TP2_LANE_CLUSTER_PX_RATIO", 0.055),
    min_lane_width_ratio=env_float("TP2_LANE_MIN_WIDTH_RATIO", 0.18),
    max_lane_width_ratio=env_float("TP2_LANE_MAX_WIDTH_RATIO", 0.72),
    max_partial_lane_width_ratio=env_float("TP2_LANE_MAX_PARTIAL_WIDTH_RATIO", 0.92),
    expected_lane_width_ratio=env_float("TP2_LANE_EXPECTED_WIDTH_RATIO", 0.38),
    preferred_corridor=os.getenv("TP2_LANE_PREFERRED_CORRIDOR", "right"),
    preferred_corridor_bonus=env_float("TP2_LANE_PREFERRED_CORRIDOR_BONUS", 1.05),
    single_line_confidence_scale=env_float("TP2_LANE_SINGLE_LINE_CONFIDENCE_SCALE", 0.58),
    stale_sec=env_float("TP2_LANE_STALE_SEC", 0.45),
    min_confidence=env_float("TP2_LANE_MIN_CONFIDENCE", 0.34),
    steering_gain=env_float("TP2_LANE_STEERING_GAIN", 2.10),
    heading_gain=env_float("TP2_LANE_HEADING_GAIN", 0.80),
    max_correction=env_float("TP2_LANE_MAX_CORRECTION", 0.75),
    smoothing_alpha=env_float("TP2_LANE_SMOOTHING_ALPHA", 0.75),
    departure_center_error=env_float("TP2_LANE_DEPARTURE_CENTER_ERROR", 0.16),
    recovery_correction_scale=env_float("TP2_LANE_RECOVERY_CORRECTION_SCALE", 1.55),
)
LANE_RECOVERY_THROTTLE = env_float("TP2_LANE_RECOVERY_THROTTLE", 0.35)
LANE_ASSIST_ACTIONS = env_csv_set(
    "TP2_LANE_ASSIST_ACTIONS",
    {"continue", "speed-30", "speed-90", "approach-stop", "confirming", "cooldown"},
)
LIDAR_CONFIG = LidarConfig(
    enabled=env_bool("TP2_LIDAR_ASSIST_ENABLED", True),
    stale_sec=env_float("TP2_LIDAR_STALE_SEC", 0.75),
    min_range_m=env_float("TP2_LIDAR_MIN_RANGE_M", 0.05),
    max_range_m=env_float("TP2_LIDAR_MAX_RANGE_M", 8.0),
    front_angle_deg=env_float("TP2_LIDAR_FRONT_ANGLE_DEG", 34.0),
    side_angle_deg=env_float("TP2_LIDAR_SIDE_ANGLE_DEG", 82.0),
    stop_distance_m=env_float("TP2_LIDAR_STOP_DISTANCE_M", 0.42),
    slow_distance_m=env_float("TP2_LIDAR_SLOW_DISTANCE_M", 0.85),
    caution_distance_m=env_float("TP2_LIDAR_CAUTION_DISTANCE_M", 1.35),
    slow_throttle=env_float("TP2_LIDAR_SLOW_THROTTLE", 0.25),
    avoidance_gain=env_float("TP2_LIDAR_AVOIDANCE_GAIN", 0.55),
    max_steering_correction=env_float("TP2_LIDAR_MAX_STEERING_CORRECTION", 0.45),
    center_deadband_m=env_float("TP2_LIDAR_CENTER_DEADBAND_M", 0.08),
    max_status_points=max(0, env_int("TP2_LIDAR_STATUS_MAX_POINTS", 720)),
)
TURN_COMPENSATION_ENABLED = env_bool("TP2_TURN_COMPENSATION_ENABLED", False)
TURN_COMPENSATION_INTERVAL_SEC = max(0.0, env_float("TP2_TURN_COMPENSATION_INTERVAL_SEC", 2.5))
TURN_COMPENSATION_DURATION_SEC = max(0.0, env_float("TP2_TURN_COMPENSATION_DURATION_SEC", 0.18))
TURN_COMPENSATION_MAGNITUDE = max(0.0, abs(env_float("TP2_TURN_COMPENSATION_MAGNITUDE", 0.20)))
TURN_COMPENSATION_ACTIONS = env_csv_set(
    "TP2_TURN_COMPENSATION_ACTIONS",
    {"continue", "speed-30", "speed-90", "confirming", "cooldown"},
)

SESSION_RECORD_DIR = Path(os.getenv("TP2_SESSION_RECORD_DIR", "/srv/tp2/frames/autonomous")).expanduser()
SESSION_RECORD_AUTOSTART = env_bool("TP2_SESSION_RECORD_AUTOSTART", False)
SESSION_RECORD_IMAGES = env_bool("TP2_SESSION_RECORD_IMAGES", True)
SESSION_RECORD_MIN_INTERVAL_SEC = env_float("TP2_SESSION_RECORD_MIN_INTERVAL_SEC", 0.45)
SESSION_RECORD_JPEG_QUALITY = min(95, max(35, env_int("TP2_SESSION_RECORD_JPEG_QUALITY", 82)))
SESSION_RECORD_VIDEO = env_bool("TP2_SESSION_RECORD_VIDEO", True)
SESSION_RECORD_VIDEO_FPS = max(1.0, env_float("TP2_SESSION_RECORD_VIDEO_FPS", 10.0))
SESSION_RECORD_CRITICAL_IMAGES = env_bool("TP2_SESSION_RECORD_CRITICAL_IMAGES", True)
SESSION_RECORD_LOW_CONF_MIN = env_float("TP2_SESSION_RECORD_LOW_CONF_MIN", 0.35)
SESSION_RECORD_LOW_CONF_MAX = env_float("TP2_SESSION_RECORD_LOW_CONF_MAX", 0.55)
SESSION_RECORD_DISAPPEAR_FRAMES = max(1, env_int("TP2_SESSION_RECORD_DISAPPEAR_FRAMES", 3))
SESSION_RECORD_TRACK_IOU = env_float("TP2_SESSION_RECORD_TRACK_IOU", 0.10)
SESSION_RECORD_TRACK_CENTER_DISTANCE = env_float("TP2_SESSION_RECORD_TRACK_CENTER_DISTANCE", 0.18)
ENABLE_SESSION_REPLAYER = env_bool("TP2_ENABLE_SESSION_REPLAYER", True)
SESSION_REPLAYER_HOST = os.getenv("TP2_SESSION_REPLAYER_HOST", "0.0.0.0")
SESSION_REPLAYER_PORT = env_int("TP2_SESSION_REPLAYER_PORT", 8090)
CONTROL_DEFAULTS_PATH = Path(
    os.getenv("TP2_CONTROL_DEFAULTS_PATH", "~/.config/tp2/coche-control-defaults.json")
).expanduser()

EXIT_EVENT = threading.Event()


def clamp(value: Any, minimum: float, maximum: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def finite_float(value: Any, *, name: str = "value") -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not math.isfinite(number):
        raise ValueError(f"invalid {name}")
    return number


def finite_bool(value: Any, *, name: str = "value") -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled", ""}:
        return False
    raise ValueError(f"invalid {name}")


def corrected_steering(steering: float, steering_trim: float | None = None) -> float:
    trim = STEERING_TRIM if steering_trim is None else finite_float(steering_trim, name="steering_trim")
    return round(clamp(float(steering) + trim, -1.0, 1.0, NEUTRAL_STEERING), 3)


RUNTIME_SETTING_RANGES: dict[str, tuple[float, float]] = {
    "steering_trim": (-1.0, 1.0),
    "manual_forward_throttle": (0.0, 1.0),
    "manual_reverse_throttle": (-1.0, 0.0),
    "manual_brake_throttle": (-1.0, 0.0),
    "crawl_throttle": (0.0, 1.0),
    "slow_throttle": (0.0, 1.0),
    "turn_throttle": (0.0, 1.0),
    "cruise_throttle": (0.0, 1.0),
    "fast_throttle": (0.0, 1.0),
    "left_steering": (-1.0, 1.0),
    "right_steering": (-1.0, 1.0),
    "stop_hold_sec": (0.0, 5.0),
    "turn_hold_sec": (0.0, 5.0),
    "cooldown_sec": (0.0, 5.0),
    "min_area_ratio": (0.0001, 0.1),
    "near_area_ratio": (0.0001, 0.25),
    "lane_recovery_throttle": (0.0, 1.0),
    "lane_steering_gain": (0.0, 5.0),
    "lane_heading_gain": (0.0, 3.0),
    "lane_max_correction": (0.0, 1.0),
    "lane_target_center_x": (0.0, 1.0),
    "lane_min_confidence": (0.0, 1.0),
    "turn_compensation_interval_sec": (0.0, 60.0),
    "turn_compensation_duration_sec": (0.0, 10.0),
    "turn_compensation_magnitude": (0.0, 2.0),
}
RUNTIME_BOOL_SETTINGS = {"turn_pulse_enabled", "lane_enabled", "turn_compensation_enabled"}
AUTONOMOUS_RUNTIME_FIELDS = {
    "crawl_throttle",
    "slow_throttle",
    "turn_throttle",
    "cruise_throttle",
    "fast_throttle",
    "left_steering",
    "right_steering",
    "stop_hold_sec",
    "turn_hold_sec",
    "cooldown_sec",
    "min_area_ratio",
    "near_area_ratio",
    "turn_pulse_enabled",
}
LANE_RUNTIME_FIELDS = {
    "lane_enabled": "enabled",
    "lane_steering_gain": "steering_gain",
    "lane_heading_gain": "heading_gain",
    "lane_max_correction": "max_correction",
    "lane_target_center_x": "target_center_x",
    "lane_min_confidence": "min_confidence",
}


def normalize_runtime_setting(name: str, value: Any) -> float | bool | None:
    if name in RUNTIME_BOOL_SETTINGS:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return None
    value_range = RUNTIME_SETTING_RANGES.get(name)
    if value_range is None:
        return None
    low, high = value_range
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(clamp(number, low, high, low), 4)


def runtime_setting_defaults() -> dict[str, Any]:
    return {
        "steering_trim": STEERING_TRIM,
        "manual_forward_throttle": MANUAL_FORWARD_THROTTLE,
        "manual_reverse_throttle": MANUAL_REVERSE_THROTTLE,
        "manual_brake_throttle": MANUAL_BRAKE_THROTTLE,
        "crawl_throttle": AUTONOMOUS_CONFIG.crawl_throttle,
        "slow_throttle": AUTONOMOUS_CONFIG.slow_throttle,
        "turn_throttle": AUTONOMOUS_CONFIG.turn_throttle,
        "cruise_throttle": AUTONOMOUS_CONFIG.cruise_throttle,
        "fast_throttle": AUTONOMOUS_CONFIG.fast_throttle,
        "left_steering": AUTONOMOUS_CONFIG.left_steering,
        "right_steering": AUTONOMOUS_CONFIG.right_steering,
        "stop_hold_sec": AUTONOMOUS_CONFIG.stop_hold_sec,
        "turn_hold_sec": AUTONOMOUS_CONFIG.turn_hold_sec,
        "turn_pulse_enabled": AUTONOMOUS_CONFIG.turn_pulse_enabled,
        "cooldown_sec": AUTONOMOUS_CONFIG.cooldown_sec,
        "min_area_ratio": AUTONOMOUS_CONFIG.min_area_ratio,
        "near_area_ratio": AUTONOMOUS_CONFIG.near_area_ratio,
        "lane_enabled": LANE_CONFIG.enabled,
        "lane_recovery_throttle": LANE_RECOVERY_THROTTLE,
        "lane_steering_gain": LANE_CONFIG.steering_gain,
        "lane_heading_gain": LANE_CONFIG.heading_gain,
        "lane_max_correction": LANE_CONFIG.max_correction,
        "lane_target_center_x": LANE_CONFIG.target_center_x,
        "lane_min_confidence": LANE_CONFIG.min_confidence,
        "turn_compensation_enabled": TURN_COMPENSATION_ENABLED,
        "turn_compensation_interval_sec": TURN_COMPENSATION_INTERVAL_SEC,
        "turn_compensation_duration_sec": TURN_COMPENSATION_DURATION_SEC,
        "turn_compensation_magnitude": TURN_COMPENSATION_MAGNITUDE,
    }


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


@dataclass
class RecorderTrack:
    track_id: int
    label: str
    first_seq: int
    last_seq: int
    hits: int
    last_prediction: dict[str, Any]
    disappeared_reported: bool = False


class CriticalFrameAnalyzer:
    def __init__(
        self,
        *,
        low_confidence_min: float,
        low_confidence_max: float,
        disappear_frames: int,
        match_iou: float,
        match_center_distance: float,
    ) -> None:
        self.low_confidence_min = low_confidence_min
        self.low_confidence_max = low_confidence_max
        self.disappear_frames = max(1, disappear_frames)
        self.match_iou = max(0.0, match_iou)
        self.match_center_distance = max(0.0, match_center_distance)
        self.next_track_id = 1
        self.tracks: dict[int, RecorderTrack] = {}

    def evaluate(
        self,
        *,
        frame_seq: int,
        frame_shape: tuple[int, ...],
        predictions: list[dict[str, Any]],
        decision: AutonomousDecision,
        operator_events: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        flags: list[dict[str, Any]] = []
        enriched: list[dict[str, Any]] = []
        matched: set[int] = set()

        for index, prediction in enumerate(predictions):
            item = dict(prediction)
            label = prediction_label(item)
            confidence = prediction_confidence(item)
            track = self._best_match(item, frame_shape, matched)

            if track is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                track = RecorderTrack(
                    track_id=track_id,
                    label=label,
                    first_seq=frame_seq,
                    last_seq=frame_seq,
                    hits=1,
                    last_prediction=dict(item),
                )
                self.tracks[track_id] = track
            else:
                if label and track.label and label != track.label and frame_seq - track.last_seq <= 1:
                    flags.append(
                        {
                            "rule": "track_class_change",
                            "track_id": track.track_id,
                            "prediction_index": index,
                            "previous_class": track.label,
                            "current_class": label,
                            "severity": "high",
                        }
                    )
                track.label = label or track.label
                track.last_seq = frame_seq
                track.hits += 1
                track.last_prediction = dict(item)
                track.disappeared_reported = False

            matched.add(track.track_id)
            item["track_id"] = track.track_id
            item["track_hits"] = track.hits
            if confidence is not None and self.low_confidence_min <= confidence <= self.low_confidence_max:
                flags.append(
                    {
                        "rule": "low_confidence_band",
                        "track_id": track.track_id,
                        "prediction_index": index,
                        "class": label,
                        "confidence": round(confidence, 4),
                        "range": [self.low_confidence_min, self.low_confidence_max],
                        "severity": "medium",
                    }
                )
            enriched.append(item)

        for track_id, track in list(self.tracks.items()):
            if track_id in matched:
                continue
            missing_frames = frame_seq - track.last_seq
            if (
                missing_frames == 1
                and track.hits < self.disappear_frames
                and not track.disappeared_reported
            ):
                flags.append(
                    {
                        "rule": "short_lived_detection",
                        "track_id": track.track_id,
                        "class": track.label,
                        "hits": track.hits,
                        "first_seq": track.first_seq,
                        "last_seq": track.last_seq,
                        "severity": "medium",
                    }
                )
                track.disappeared_reported = True
            if missing_frames > max(self.disappear_frames, 6):
                self.tracks.pop(track_id, None)

        decision_status = decision.to_status()
        if decision.action == "ambiguous" or decision.state == "ambiguous":
            flags.append(
                {
                    "rule": "ambiguous_decision",
                    "action": decision.action,
                    "state": decision.state,
                    "reason": decision.reason,
                    "severity": "high",
                }
            )

        for event in operator_events:
            if event.get("type") in {"manual_override", "manual_override_attempt"}:
                flags.append(
                    {
                        "rule": "operator_override",
                        "event": event,
                        "severity": "high",
                    }
                )

        return enriched, dedupe_flags(flags, decision_status)

    def _best_match(
        self,
        prediction: dict[str, Any],
        frame_shape: tuple[int, ...],
        matched: set[int],
    ) -> RecorderTrack | None:
        best: tuple[float, RecorderTrack] | None = None
        for track in self.tracks.values():
            if track.track_id in matched:
                continue
            overlap = prediction_iou(track.last_prediction, prediction)
            center_distance = prediction_center_distance(
                track.last_prediction,
                prediction,
                frame_shape,
            )
            if overlap < self.match_iou and center_distance > self.match_center_distance:
                continue
            score = overlap + max(0.0, 1.0 - center_distance)
            if best is None or score > best[0]:
                best = (score, track)
        return None if best is None else best[1]


class SessionRecorder:
    def __init__(
        self,
        root: Path,
        *,
        autostart: bool,
        save_images: bool,
        min_interval_sec: float,
        jpeg_quality: int,
        save_video: bool,
        video_fps: float,
        save_critical_images: bool,
    ) -> None:
        self.root = root
        self.save_images = save_images
        self.min_interval_sec = max(0.0, min_interval_sec)
        self.jpeg_quality = jpeg_quality
        self.save_video = save_video
        self.video_fps = max(1.0, video_fps)
        self.save_critical_images = save_critical_images
        self.lock = threading.RLock()
        self.enabled = False
        self.session_dir: Path | None = None
        self.images_dir: Path | None = None
        self.critical_dir: Path | None = None
        self.manifest_path: Path | None = None
        self.labels_path: Path | None = None
        self.critical_path: Path | None = None
        self.video_path: Path | None = None
        self.video_writer: cv2.VideoWriter | None = None
        self.video_size: tuple[int, int] | None = None
        self.started_at: float | None = None
        self.last_record_at = 0.0
        self.records = 0
        self.images = 0
        self.critical_records = 0
        self.video_frames = 0
        self.last_error: str | None = None
        self.critical_analyzer = self._new_critical_analyzer()
        if autostart:
            self.start()

    def start(self) -> dict[str, Any]:
        with self.lock:
            if self.enabled:
                return self.snapshot()
            self._release_video_writer()
            self.critical_analyzer = self._new_critical_analyzer()
            session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.session_dir = self.root / session_id
            self.images_dir = self.session_dir / "images"
            self.critical_dir = self.session_dir / "critical"
            self.manifest_path = self.session_dir / "manifest.jsonl"
            self.labels_path = self.session_dir / "labels.jsonl"
            self.critical_path = self.session_dir / "critical.jsonl"
            self.video_path = self.session_dir / "session.mp4"
            self.video_size = None
            try:
                self.session_dir.mkdir(parents=True, exist_ok=True)
                if self.save_images:
                    self.images_dir.mkdir(parents=True, exist_ok=True)
                if self.save_critical_images:
                    self.critical_dir.mkdir(parents=True, exist_ok=True)
                session_meta = {
                    "schema_version": 2,
                    "session_id": session_id,
                    "started_at": datetime.now().isoformat(timespec="milliseconds"),
                    "recording": {
                        "images": self.save_images,
                        "video": self.save_video,
                        "video_fps": self.video_fps,
                        "critical_images": self.save_critical_images,
                        "min_interval_sec": self.min_interval_sec,
                    },
                    "critical_rules": {
                        "low_confidence": [
                            SESSION_RECORD_LOW_CONF_MIN,
                            SESSION_RECORD_LOW_CONF_MAX,
                        ],
                        "short_lived_detection_frames": SESSION_RECORD_DISAPPEAR_FRAMES,
                        "track_class_change": True,
                        "ambiguous_decision": True,
                        "operator_override": True,
                    },
                }
                (self.session_dir / "session.json").write_text(
                    json.dumps(session_meta, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8",
                )
                (self.session_dir / "README.txt").write_text(
                    "TP2 autonomous session capture.\n"
                    "manifest.jsonl maps frame_seq to raw image, annotated video frame, "
                    "Roboflow predictions, critical flags, autonomous decision, and control.\n"
                    "labels.jsonl contains model-estimated labels for offline review; "
                    "labels_reviewed.json is written by session_replayer.py.\n",
                    encoding="utf-8",
                )
                self.enabled = True
                self.started_at = wall_time()
                self.last_record_at = 0.0
                self.records = 0
                self.images = 0
                self.critical_records = 0
                self.video_frames = 0
                self.last_error = None
            except OSError as exc:
                self.enabled = False
                self.last_error = str(exc)
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            self.enabled = False
            self._release_video_writer()
            return self.snapshot()

    def close(self) -> None:
        with self.lock:
            self._release_video_writer()

    def set_enabled(self, enabled: bool) -> dict[str, Any]:
        return self.start() if enabled else self.stop()

    def record(
        self,
        *,
        frame: np.ndarray,
        frame_seq: int,
        predictions: list[dict[str, Any]],
        inference_payload: Any,
        decision: AutonomousDecision,
        inference_latency_ms: int | None,
        inference_backend: dict[str, Any],
        control: dict[str, Any],
        operator_events: list[dict[str, Any]],
    ) -> None:
        with self.lock:
            if not self.enabled or self.session_dir is None or self.manifest_path is None:
                return
            now = wall_time()
            if now - self.last_record_at < self.min_interval_sec:
                return
            self.last_record_at = now

            enriched_predictions, critical_flags = self.critical_analyzer.evaluate(
                frame_seq=frame_seq,
                frame_shape=frame.shape,
                predictions=predictions,
                decision=decision,
                operator_events=operator_events,
            )
            labels = build_label_candidates(enriched_predictions, frame.shape)

            image_rel: str | None = None
            if self.save_images and self.images_dir is not None:
                image_rel = f"images/frame_{frame_seq:08d}.jpg"
                image_path = self.session_dir / image_rel
                try:
                    ok = cv2.imwrite(
                        str(image_path),
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                    )
                    if not ok:
                        raise RuntimeError("cv2.imwrite returned false")
                    self.images += 1
                except Exception as exc:
                    self.last_error = f"image: {exc}"
                    image_rel = None

            annotated = draw_recording_overlay(
                frame,
                enriched_predictions,
                decision=decision,
                critical_flags=critical_flags,
            )
            video_info = self._write_video_frame(annotated)

            critical_rel: str | None = None
            if critical_flags:
                self.critical_records += 1
                if self.save_critical_images and self.critical_dir is not None:
                    critical_rel = f"critical/frame_{frame_seq:08d}.jpg"
                    critical_path = self.session_dir / critical_rel
                    try:
                        cv2.imwrite(
                            str(critical_path),
                            annotated,
                            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                        )
                    except Exception as exc:
                        self.last_error = f"critical-image: {exc}"

            item = {
                "schema_version": 2,
                "ts": round(now, 3),
                "iso_time": datetime.now().isoformat(timespec="milliseconds"),
                "frame_seq": frame_seq,
                "record_index": self.records,
                "image": image_rel,
                "critical_image": critical_rel,
                "video": video_info,
                "predictions": sanitize_predictions(enriched_predictions),
                "labels": labels,
                "raw_prediction_count": len(predictions),
                "inference_payload": summarize_payload(inference_payload),
                "inference_latency_ms": inference_latency_ms,
                "inference_backend": inference_backend,
                "autonomy": decision.to_status(),
                "control": control,
                "operator_events": operator_events,
                "critical": {
                    "is_critical": bool(critical_flags),
                    "flags": critical_flags,
                },
                "roboflow_retrain_note": "candidate-estimates-not-ground-truth",
            }
            try:
                self._append_jsonl(self.manifest_path, item)
                if self.labels_path is not None:
                    self._append_jsonl(
                        self.labels_path,
                        {
                            "frame_seq": frame_seq,
                            "image": image_rel,
                            "labels": labels,
                            "source": "model-candidate",
                            "reviewed": False,
                        },
                    )
                if critical_flags and self.critical_path is not None:
                    self._append_jsonl(
                        self.critical_path,
                        {
                            "frame_seq": frame_seq,
                            "image": image_rel,
                            "critical_image": critical_rel,
                            "flags": critical_flags,
                            "autonomy": decision.to_status(),
                            "operator_events": operator_events,
                        },
                    )
                self.records += 1
                self.last_error = None
            except OSError as exc:
                self.last_error = f"manifest: {exc}"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            now = wall_time()
            return {
                "enabled": self.enabled,
                "root": str(self.root),
                "session_dir": None if self.session_dir is None else str(self.session_dir),
                "records": self.records,
                "images": self.images,
                "critical_records": self.critical_records,
                "video": {
                    "enabled": self.save_video,
                    "path": None if self.video_path is None else str(self.video_path),
                    "frames": self.video_frames,
                    "fps": self.video_fps,
                },
                "age_sec": None if self.started_at is None else rounded(now - self.started_at),
                "min_interval_sec": self.min_interval_sec,
                "save_images": self.save_images,
                "last_error": self.last_error,
            }

    def _new_critical_analyzer(self) -> CriticalFrameAnalyzer:
        return CriticalFrameAnalyzer(
            low_confidence_min=SESSION_RECORD_LOW_CONF_MIN,
            low_confidence_max=SESSION_RECORD_LOW_CONF_MAX,
            disappear_frames=SESSION_RECORD_DISAPPEAR_FRAMES,
            match_iou=SESSION_RECORD_TRACK_IOU,
            match_center_distance=SESSION_RECORD_TRACK_CENTER_DISTANCE,
        )

    def _write_video_frame(self, frame: np.ndarray) -> dict[str, Any] | None:
        if not self.save_video or self.video_path is None:
            return None
        try:
            h, w = frame.shape[:2]
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_size = (w, h)
                self.video_writer = cv2.VideoWriter(
                    str(self.video_path),
                    fourcc,
                    self.video_fps,
                    self.video_size,
                )
                if not self.video_writer.isOpened():
                    self.video_writer = None
                    raise RuntimeError("cv2.VideoWriter could not open session.mp4")
            if self.video_size is not None and (w, h) != self.video_size:
                frame = cv2.resize(frame, self.video_size, interpolation=cv2.INTER_AREA)
            frame_index = self.video_frames
            self.video_writer.write(frame)
            self.video_frames += 1
            return {
                "path": "session.mp4",
                "frame_index": frame_index,
                "fps": self.video_fps,
            }
        except Exception as exc:
            self.last_error = f"video: {exc}"
            return None

    def _release_video_writer(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    @staticmethod
    def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n")


def prediction_label(prediction: dict[str, Any]) -> str:
    return str(prediction.get("class") or prediction.get("class_name") or "").strip()


def prediction_confidence(prediction: dict[str, Any]) -> float | None:
    value = prediction.get("confidence")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def prediction_box(prediction: dict[str, Any]) -> tuple[float, float, float, float] | None:
    try:
        x = float(prediction.get("x"))
        y = float(prediction.get("y"))
        w = float(prediction.get("width"))
        h = float(prediction.get("height"))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def prediction_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    box_a = prediction_box(a)
    box_b = prediction_box(b)
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def prediction_center_distance(
    a: dict[str, Any],
    b: dict[str, Any],
    frame_shape: tuple[int, ...],
) -> float:
    try:
        ax = float(a.get("x"))
        ay = float(a.get("y"))
        bx = float(b.get("x"))
        by = float(b.get("y"))
    except (TypeError, ValueError):
        return 1.0
    frame_h = max(1, int(frame_shape[0])) if len(frame_shape) > 0 else 1
    frame_w = max(1, int(frame_shape[1])) if len(frame_shape) > 1 else 1
    return float(np.hypot((ax - bx) / frame_w, (ay - by) / frame_h))


def dedupe_flags(
    flags: list[dict[str, Any]],
    decision_status: dict[str, Any],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for flag in flags:
        key = json.dumps(
            {
                "rule": flag.get("rule"),
                "track_id": flag.get("track_id"),
                "prediction_index": flag.get("prediction_index"),
                "event_seq": (flag.get("event") or {}).get("seq"),
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        item = dict(flag)
        item.setdefault("decision_action", decision_status.get("action"))
        result.append(item)
    return result


def build_label_candidates(
    predictions: list[dict[str, Any]],
    frame_shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    frame_h = max(1, int(frame_shape[0])) if len(frame_shape) > 0 else 1
    frame_w = max(1, int(frame_shape[1])) if len(frame_shape) > 1 else 1
    labels: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        box = prediction_box(prediction)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        labels.append(
            {
                "index": index,
                "track_id": prediction.get("track_id"),
                "class": prediction_label(prediction),
                "confidence": prediction_confidence(prediction),
                "bbox_xyxy": [
                    round(max(0.0, min(frame_w, x1)), 2),
                    round(max(0.0, min(frame_h, y1)), 2),
                    round(max(0.0, min(frame_w, x2)), 2),
                    round(max(0.0, min(frame_h, y2)), 2),
                ],
                "bbox_normalized_xyxy": [
                    round(max(0.0, min(1.0, x1 / frame_w)), 6),
                    round(max(0.0, min(1.0, y1 / frame_h)), 6),
                    round(max(0.0, min(1.0, x2 / frame_w)), 6),
                    round(max(0.0, min(1.0, y2 / frame_h)), 6),
                ],
                "status": "candidate",
            }
        )
    return labels


def draw_recording_overlay(
    frame: np.ndarray,
    predictions: list[dict[str, Any]],
    *,
    decision: AutonomousDecision,
    critical_flags: list[dict[str, Any]],
) -> np.ndarray:
    output = draw_predictions_on_image(
        frame,
        predictions,
        min_confidence=0.0,
    )
    h, w = output.shape[:2]

    for prediction in predictions:
        box = prediction_box(prediction)
        if box is None:
            continue
        x1, y1, _x2, _y2 = [int(round(v)) for v in box]
        track_id = prediction.get("track_id")
        if track_id is None:
            continue
        text = f"#{track_id}"
        cv2.putText(
            output,
            text,
            (max(0, x1), min(h - 8, max(14, y1 + 16))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            text,
            (max(0, x1), min(h - 8, max(14, y1 + 16))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    banner = f"auto={decision.action} state={decision.state}"
    if critical_flags:
        rules = ",".join(str(flag.get("rule", "?")) for flag in critical_flags[:3])
        banner += f" critical={rules}"
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), (8, 8, 10), -1)
    cv2.addWeighted(overlay, 0.70, output, 0.30, 0, output)
    cv2.putText(
        output,
        banner[:120],
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (248, 248, 250) if not critical_flags else (98, 190, 255),
        1,
        cv2.LINE_AA,
    )
    return output


class ReplayerManager:
    def __init__(
        self,
        record_root: Path,
        *,
        enabled: bool,
        host: str,
        port: int,
    ) -> None:
        self.record_root = record_root
        self.enabled = enabled
        self.host = host
        self.port = port
        self.lock = threading.RLock()
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.last_error: str | None = None

    def start(self) -> dict[str, Any]:
        with self.lock:
            if not self.enabled:
                self.last_error = "session replayer disabled"
                return self.snapshot()
            if self.server is not None:
                return self.snapshot()
            try:
                catalog = SessionCatalog(self.record_root)
                ReplayerHandler.catalog = catalog
                server = ThreadingHTTPServer((self.host, self.port), ReplayerHandler)
                thread = threading.Thread(
                    target=server.serve_forever,
                    daemon=True,
                    name="session-replayer",
                )
                thread.start()
                self.server = server
                self.thread = thread
                self.last_error = None
            except OSError as exc:
                self.server = None
                self.thread = None
                self.last_error = str(exc)
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            server = self.server
            self.server = None
            self.thread = None
        if server is not None:
            def shutdown() -> None:
                server.shutdown()
                server.server_close()

            threading.Thread(target=shutdown, daemon=True).start()
        return self.snapshot()

    def snapshot(self, *, public_host: str | None = None) -> dict[str, Any]:
        with self.lock:
            active = self.server is not None
            host = public_host or self.host
            if host in {"", "0.0.0.0", "::"}:
                host = "127.0.0.1"
            return {
                "enabled": self.enabled,
                "active": active,
                "host": self.host,
                "port": self.port,
                "url": f"http://{host}:{self.port}/",
                "record_root": str(self.record_root),
                "last_error": self.last_error,
            }


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
        self.settings_path = CONTROL_DEFAULTS_PATH
        self.settings_saved_at: str | None = None
        self.settings_last_error: str | None = None
        self.steering_trim = STEERING_TRIM
        self.manual_forward_throttle = MANUAL_FORWARD_THROTTLE
        self.manual_reverse_throttle = MANUAL_REVERSE_THROTTLE
        self.manual_brake_throttle = MANUAL_BRAKE_THROTTLE
        self.autonomous_config = AUTONOMOUS_CONFIG
        self.lane_config = LANE_CONFIG
        self.lane_recovery_throttle = LANE_RECOVERY_THROTTLE
        self.lane_assist_actions = set(LANE_ASSIST_ACTIONS)

        self.lane_detector = LaneDetector(self.lane_config)
        self.lane_guidance: LaneGuidance | None = None
        self.lane_guidance_at: float | None = None
        self.lane_frames = 0
        self.lane_errors = 0
        self.lane_error: str | None = None
        self.lane_assist_active = False
        self.lane_assist_correction = 0.0
        self.lane_assist_reason = "not-evaluated"

        self.lidar_config = LIDAR_CONFIG
        self.lidar_scan: LidarScan | None = None
        self.lidar_safety = analyze_lidar_scan(None, config=self.lidar_config, now=wall_time())
        self.lidar_frames = 0
        self.lidar_errors = 0
        self.lidar_error: str | None = None
        self.lidar_assist_active = False
        self.lidar_assist_reason = "not-evaluated"

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
        self.steering_trim = STEERING_TRIM
        self.throttle = NEUTRAL_THROTTLE
        self.control_updated_at = wall_time()
        self.control_seq = 0
        self.operator_event_seq = 0
        self.pending_operator_events: list[dict[str, Any]] = []
        self.drive_mode = normalize_drive_mode(DEFAULT_DRIVE_MODE)
        self.autonomous_controller = AutonomousController(self.autonomous_config)
        self.autonomous_decision = AutonomousDecision(
            active=False,
            steering=NEUTRAL_STEERING,
            throttle=NEUTRAL_THROTTLE,
            raw_steering=NEUTRAL_STEERING,
            raw_throttle=NEUTRAL_THROTTLE,
            action="safe-neutral",
            state="safe",
            reason="not-evaluated",
            target=None,
            candidates=(),
        )
        self.turn_compensation_enabled = TURN_COMPENSATION_ENABLED
        self.turn_compensation_interval_sec = TURN_COMPENSATION_INTERVAL_SEC
        self.turn_compensation_duration_sec = min(
            TURN_COMPENSATION_DURATION_SEC,
            TURN_COMPENSATION_INTERVAL_SEC,
        ) if TURN_COMPENSATION_INTERVAL_SEC > 0 else TURN_COMPENSATION_DURATION_SEC
        self.turn_compensation_magnitude = TURN_COMPENSATION_MAGNITUDE
        self.turn_compensation_last_pulse_at: float | None = self.started_at
        self.turn_compensation_active_until = 0.0
        self.turn_compensation_active = False
        self.turn_compensation_applied_correction = 0.0
        self.turn_compensation_reason = "disabled" if not TURN_COMPENSATION_ENABLED else "waiting"
        if self.drive_mode == "autonomous":
            self.control_source = "autonomous"

        self._load_saved_control_defaults_locked()

        self.recorder = SessionRecorder(
            SESSION_RECORD_DIR,
            autostart=SESSION_RECORD_AUTOSTART,
            save_images=SESSION_RECORD_IMAGES,
            min_interval_sec=SESSION_RECORD_MIN_INTERVAL_SEC,
            jpeg_quality=SESSION_RECORD_JPEG_QUALITY,
            save_video=SESSION_RECORD_VIDEO,
            video_fps=SESSION_RECORD_VIDEO_FPS,
            save_critical_images=SESSION_RECORD_CRITICAL_IMAGES,
        )
        self.replayer = ReplayerManager(
            SESSION_RECORD_DIR,
            enabled=ENABLE_SESSION_REPLAYER,
            host=SESSION_REPLAYER_HOST,
            port=SESSION_REPLAYER_PORT,
        )

        self.web_stream_clients = 0
        self.web_control_posts = 0

    def _load_saved_control_defaults_locked(self) -> None:
        if not self.settings_path.exists():
            return
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
            values = payload.get("values", payload)
            if not isinstance(values, dict):
                raise ValueError("settings values must be an object")
            self._apply_runtime_settings_locked(values, rebuild=True)
            saved_at = payload.get("saved_at")
            self.settings_saved_at = str(saved_at) if saved_at else None
            self.settings_last_error = None
        except Exception as exc:
            self.settings_last_error = f"load defaults: {exc}"

    def _runtime_settings_values_locked(self) -> dict[str, Any]:
        return {
            "steering_trim": round(self.steering_trim, 4),
            "manual_forward_throttle": round(self.manual_forward_throttle, 4),
            "manual_reverse_throttle": round(self.manual_reverse_throttle, 4),
            "manual_brake_throttle": round(self.manual_brake_throttle, 4),
            "crawl_throttle": round(self.autonomous_config.crawl_throttle, 4),
            "slow_throttle": round(self.autonomous_config.slow_throttle, 4),
            "turn_throttle": round(self.autonomous_config.turn_throttle, 4),
            "cruise_throttle": round(self.autonomous_config.cruise_throttle, 4),
            "fast_throttle": round(self.autonomous_config.fast_throttle, 4),
            "left_steering": round(self.autonomous_config.left_steering, 4),
            "right_steering": round(self.autonomous_config.right_steering, 4),
            "stop_hold_sec": round(self.autonomous_config.stop_hold_sec, 4),
            "turn_hold_sec": round(self.autonomous_config.turn_hold_sec, 4),
            "turn_pulse_enabled": bool(self.autonomous_config.turn_pulse_enabled),
            "cooldown_sec": round(self.autonomous_config.cooldown_sec, 4),
            "min_area_ratio": round(self.autonomous_config.min_area_ratio, 4),
            "near_area_ratio": round(self.autonomous_config.near_area_ratio, 4),
            "lane_enabled": bool(self.lane_config.enabled),
            "lane_recovery_throttle": round(self.lane_recovery_throttle, 4),
            "lane_steering_gain": round(self.lane_config.steering_gain, 4),
            "lane_heading_gain": round(self.lane_config.heading_gain, 4),
            "lane_max_correction": round(self.lane_config.max_correction, 4),
            "lane_target_center_x": round(self.lane_config.target_center_x, 4),
            "lane_min_confidence": round(self.lane_config.min_confidence, 4),
            "turn_compensation_enabled": bool(self.turn_compensation_enabled),
            "turn_compensation_interval_sec": round(self.turn_compensation_interval_sec, 4),
            "turn_compensation_duration_sec": round(self.turn_compensation_duration_sec, 4),
            "turn_compensation_magnitude": round(self.turn_compensation_magnitude, 4),
        }

    def _apply_runtime_settings_locked(self, values: dict[str, Any], *, rebuild: bool) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for name, value in values.items():
            clean = normalize_runtime_setting(str(name), value)
            if clean is not None:
                normalized[str(name)] = clean

        if not normalized:
            return self._runtime_settings_values_locked()

        if "steering_trim" in normalized:
            self.steering_trim = float(normalized["steering_trim"])
        if "manual_forward_throttle" in normalized:
            self.manual_forward_throttle = float(normalized["manual_forward_throttle"])
        if "manual_reverse_throttle" in normalized:
            self.manual_reverse_throttle = float(normalized["manual_reverse_throttle"])
        if "manual_brake_throttle" in normalized:
            self.manual_brake_throttle = float(normalized["manual_brake_throttle"])
        if "lane_recovery_throttle" in normalized:
            self.lane_recovery_throttle = float(normalized["lane_recovery_throttle"])

        autonomous_updates = {
            key: normalized[key]
            for key in AUTONOMOUS_RUNTIME_FIELDS
            if key in normalized
        }
        if autonomous_updates:
            self.autonomous_config = replace(self.autonomous_config, **autonomous_updates)
            if rebuild:
                self.autonomous_controller = AutonomousController(self.autonomous_config)

        lane_updates = {
            target: normalized[source]
            for source, target in LANE_RUNTIME_FIELDS.items()
            if source in normalized
        }
        if lane_updates:
            self.lane_config = replace(self.lane_config, **lane_updates)
            if rebuild:
                self.lane_detector = LaneDetector(self.lane_config)
                self.lane_guidance = None
                self.lane_guidance_at = None
                self.lane_assist_active = False
                self.lane_assist_correction = 0.0
                self.lane_assist_reason = "settings-updated"

        turn_compensation_fields = {
            "turn_compensation_enabled",
            "turn_compensation_interval_sec",
            "turn_compensation_duration_sec",
            "turn_compensation_magnitude",
        }
        if turn_compensation_fields.intersection(normalized):
            enabled_value = bool(normalized.get("turn_compensation_enabled", self.turn_compensation_enabled))
            interval_value = float(
                normalized.get(
                    "turn_compensation_interval_sec",
                    self.turn_compensation_interval_sec,
                )
            )
            duration_value = float(
                normalized.get(
                    "turn_compensation_duration_sec",
                    self.turn_compensation_duration_sec,
                )
            )
            magnitude_value = abs(
                float(
                    normalized.get(
                        "turn_compensation_magnitude",
                        self.turn_compensation_magnitude,
                    )
                )
            )
            if interval_value > 0.0:
                duration_value = min(duration_value, interval_value)
            self.turn_compensation_enabled = enabled_value
            self.turn_compensation_interval_sec = round(max(0.0, interval_value), 3)
            self.turn_compensation_duration_sec = round(max(0.0, duration_value), 3)
            self.turn_compensation_magnitude = round(clamp(magnitude_value, 0.0, 2.0, 0.0), 3)
            now = wall_time()
            self.turn_compensation_last_pulse_at = now
            self.turn_compensation_active_until = 0.0
            self.turn_compensation_active = False
            self.turn_compensation_applied_correction = 0.0
            self.turn_compensation_reason = "waiting" if enabled_value else "disabled"

        if rebuild and autonomous_updates and self.drive_mode == "autonomous":
            self.autonomous_controller.filter.reset()
            self._apply_autonomous_control_locked()
        self.control_seq += 1
        return self._runtime_settings_values_locked()

    def update_runtime_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_values = payload.get("values", payload)
        if not isinstance(raw_values, dict):
            raise ValueError("settings payload must be an object")
        with self.lock:
            values = self._apply_runtime_settings_locked(raw_values, rebuild=True)
            self.settings_last_error = None
            return self.settings_snapshot_locked(values=values)

    def save_current_settings_as_defaults(self) -> dict[str, Any]:
        with self.lock:
            values = self._runtime_settings_values_locked()
            payload = {
                "schema_version": 1,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "values": values,
            }
            try:
                self.settings_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.settings_path.with_suffix(self.settings_path.suffix + ".tmp")
                tmp.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8",
                )
                tmp.replace(self.settings_path)
                self.settings_saved_at = payload["saved_at"]
                self.settings_last_error = None
            except Exception as exc:
                self.settings_last_error = f"save defaults: {exc}"
            return self.settings_snapshot_locked(values=values)

    def settings_snapshot_locked(self, *, values: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "path": str(self.settings_path),
            "persisted": self.settings_path.exists(),
            "saved_at": self.settings_saved_at,
            "last_error": self.settings_last_error,
            "defaults": runtime_setting_defaults(),
            "values": values or self._runtime_settings_values_locked(),
            "ranges": {
                key: {"min": bounds[0], "max": bounds[1]}
                for key, bounds in RUNTIME_SETTING_RANGES.items()
            },
            "bools": sorted(RUNTIME_BOOL_SETTINGS),
        }

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

    def update_lidar(self, value: Any) -> None:
        now = wall_time()
        try:
            scan = normalize_lidar_payload(value, config=self.lidar_config, received_at=now)
            safety = analyze_lidar_scan(scan, config=self.lidar_config, now=now)
        except Exception as exc:
            with self.lock:
                self.lidar_errors += 1
                self.lidar_error = str(exc)[:240]
                self.last_packet_error = f"lidar: {self.lidar_error}"
            return
        with self.lock:
            self.lidar_scan = scan
            self.lidar_safety = safety
            self.lidar_frames += 1
            self.lidar_error = None
            if self.drive_mode == "autonomous":
                self._apply_autonomous_control_locked()

    def update_lidar_from_telemetry(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        lidar_payload = None
        for key in ("lidar", "lidar_scan", "scan", "ranges", "points"):
            if key in value:
                lidar_payload = value if key in {"ranges", "points"} else value[key]
                break
        if lidar_payload is None:
            return False
        self.update_lidar(lidar_payload)
        return True

    def update_frame(self, frame: np.ndarray) -> int:
        now = wall_time()
        lane_guidance: LaneGuidance | None = None
        lane_error: str | None = None
        if self.lane_config.enabled:
            try:
                lane_guidance = self.lane_detector.detect(frame, now=now)
            except Exception as exc:
                lane_error = str(exc)[:240]
        with self.frame_cond:
            self.latest_frame = frame
            self.latest_frame_seq += 1
            self.latest_frame_at = now
            if self.lane_config.enabled:
                if lane_guidance is not None:
                    self.lane_guidance = lane_guidance
                    self.lane_guidance_at = now
                    self.lane_frames += 1
                    self.lane_error = None
                elif lane_error is not None:
                    self.lane_errors += 1
                    self.lane_error = lane_error
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
        *,
        frame: np.ndarray | None = None,
        inference_payload: Any = None,
    ) -> None:
        with self.lock:
            self.predictions = predictions
            self.predictions_seq = seq
            self.predictions_at = wall_time()
            self.inference_status = "ready"
            self.inference_error = None
            self.inference_latency_ms = latency_ms
            self.inference_frames += 1
            decision = self._evaluate_autonomous_locked()
            control = self.control_snapshot_locked()
            backend = dict(self.inference_backend)
            operator_events = self._consume_operator_events_locked()

        if frame is not None:
            self.recorder.record(
                frame=frame,
                frame_seq=seq,
                predictions=predictions,
                inference_payload=inference_payload,
                decision=decision,
                inference_latency_ms=latency_ms,
                inference_backend=backend,
                control=control,
                operator_events=operator_events,
            )

    def record_frame_without_inference(self, seq: int, frame: np.ndarray) -> None:
        with self.lock:
            if ENABLE_INFERENCE and self.inference_status not in {"disabled", "offline", "error"}:
                return
            decision = self._evaluate_autonomous_locked()
            control = self.control_snapshot_locked()
            backend = dict(self.inference_backend)
            status = self.inference_status
            error = self.inference_error
            operator_events = self._consume_operator_events_locked()

        self.recorder.record(
            frame=frame,
            frame_seq=seq,
            predictions=[],
            inference_payload={"status": status, "error": error},
            decision=decision,
            inference_latency_ms=None,
            inference_backend=backend,
            control=control,
            operator_events=operator_events,
        )

    def _evaluate_autonomous_locked(self) -> AutonomousDecision:
        now = wall_time()
        frame_shape = None if self.latest_frame is None else self.latest_frame.shape
        decision = self.autonomous_controller.decide(
            list(self.predictions),
            frame_shape=frame_shape,
            now=now,
            frame_time=self.latest_frame_at,
            predictions_time=self.predictions_at,
            prediction_seq=self.predictions_seq,
        )
        decision = self._apply_lane_assist_locked(decision, now)
        decision = self._apply_turn_compensation_locked(decision, now)
        decision = self._apply_lidar_safety_locked(decision, now)
        self.autonomous_decision = decision
        return decision

    def _apply_lane_assist_locked(
        self,
        decision: AutonomousDecision,
        now: float,
    ) -> AutonomousDecision:
        self.lane_assist_active = False
        self.lane_assist_correction = 0.0

        lane_config = self.lane_config
        if not lane_config.enabled:
            self.lane_assist_reason = "disabled"
            return decision
        if self.drive_mode != "autonomous":
            self.lane_assist_reason = "manual-mode"
            return decision
        if not decision.active:
            self.lane_assist_reason = f"autonomy-{decision.reason}"
            return decision
        if decision.throttle <= max(0.05, NEUTRAL_THROTTLE + 0.02):
            self.lane_assist_reason = "not-moving-forward"
            return decision
        if decision.action not in self.lane_assist_actions:
            self.lane_assist_reason = f"action-{decision.action}"
            return decision

        guidance = self.current_lane_guidance_locked(now=now)
        if guidance is None:
            self.lane_assist_reason = "no-lane"
            return decision
        if not guidance.is_usable(lane_config):
            self.lane_assist_reason = f"lane-unusable:{guidance.reason}"
            return decision

        correction = clamp(guidance.correction, -lane_config.max_correction, lane_config.max_correction, 0.0)
        steering = round(clamp(decision.steering + correction, -1.0, 1.0, NEUTRAL_STEERING), 3)
        raw_base = decision.raw_steering if decision.raw_steering is not None else decision.steering
        raw_steering = round(clamp(raw_base + correction, -1.0, 1.0, NEUTRAL_STEERING), 3)
        throttle = decision.throttle
        raw_throttle = decision.raw_throttle
        recovery = abs(guidance.center_error) >= lane_config.departure_center_error
        if recovery:
            throttle = round(clamp(min(decision.throttle, self.lane_recovery_throttle), 0.0, 1.0, NEUTRAL_THROTTLE), 3)
            if raw_throttle is not None:
                raw_throttle = round(clamp(min(raw_throttle, self.lane_recovery_throttle), 0.0, 1.0, NEUTRAL_THROTTLE), 3)
        self.lane_assist_active = True
        self.lane_assist_correction = round(correction, 3)
        self.lane_assist_reason = f"{guidance.source}:{guidance.reason}"
        if recovery:
            self.lane_assist_reason += ":recovery"
        return replace(
            decision,
            steering=steering,
            throttle=throttle,
            raw_steering=raw_steering,
            raw_throttle=raw_throttle,
            reason=f"{decision.reason};lane={guidance.source}:{correction:+.3f}{':recovery' if recovery else ''}",
        )

    def _apply_turn_compensation_locked(
        self,
        decision: AutonomousDecision,
        now: float,
    ) -> AutonomousDecision:
        self.turn_compensation_active = False
        self.turn_compensation_applied_correction = 0.0

        if not self.turn_compensation_enabled:
            self.turn_compensation_reason = "disabled"
            return decision
        if self.drive_mode != "autonomous":
            self.turn_compensation_reason = "manual-mode"
            return decision
        if not decision.active:
            self.turn_compensation_reason = f"autonomy-{decision.reason}"
            return decision
        if decision.throttle <= max(0.05, NEUTRAL_THROTTLE + 0.02):
            self.turn_compensation_reason = "not-moving-forward"
            return decision
        if decision.action not in TURN_COMPENSATION_ACTIONS:
            self.turn_compensation_reason = f"action-{decision.action}"
            return decision
        if self.turn_compensation_interval_sec <= 0.0:
            self.turn_compensation_reason = "interval-disabled"
            return decision
        if self.turn_compensation_duration_sec <= 0.0:
            self.turn_compensation_reason = "duration-disabled"
            return decision
        if self.turn_compensation_magnitude <= 0.0:
            self.turn_compensation_reason = "magnitude-zero"
            return decision

        if self.turn_compensation_last_pulse_at is None:
            self.turn_compensation_last_pulse_at = now
            self.turn_compensation_reason = "waiting"
            return decision

        if now >= self.turn_compensation_active_until:
            elapsed = now - self.turn_compensation_last_pulse_at
            if elapsed >= self.turn_compensation_interval_sec:
                self.turn_compensation_last_pulse_at = now
                self.turn_compensation_active_until = now + self.turn_compensation_duration_sec
            else:
                self.turn_compensation_reason = "between-pulses"
                return decision

        if now >= self.turn_compensation_active_until:
            self.turn_compensation_reason = "between-pulses"
            return decision

        correction = -abs(self.turn_compensation_magnitude)
        steering = round(clamp(decision.steering + correction, -1.0, 1.0, NEUTRAL_STEERING), 3)
        raw_base = decision.raw_steering if decision.raw_steering is not None else decision.steering
        raw_steering = round(clamp(raw_base + correction, -1.0, 1.0, NEUTRAL_STEERING), 3)
        self.turn_compensation_active = True
        self.turn_compensation_applied_correction = round(correction, 3)
        self.turn_compensation_reason = "pulse-right"
        return replace(
            decision,
            steering=steering,
            raw_steering=raw_steering,
            reason=f"{decision.reason};turn-comp={correction:+.3f}",
        )

    def _apply_lidar_safety_locked(
        self,
        decision: AutonomousDecision,
        now: float,
    ) -> AutonomousDecision:
        safety = analyze_lidar_scan(self.lidar_scan, config=self.lidar_config, now=now)
        self.lidar_safety = safety
        self.lidar_assist_active = False

        if not self.lidar_config.enabled:
            self.lidar_assist_reason = "disabled"
            return decision
        if self.drive_mode != "autonomous":
            self.lidar_assist_reason = "manual-mode"
            return decision
        if not decision.active:
            self.lidar_assist_reason = f"autonomy-{decision.reason}"
            return decision
        if safety.status in {"searching", "stale", "clear"}:
            self.lidar_assist_reason = safety.reason
            return decision
        if safety.status == "stop":
            self.lidar_assist_active = True
            self.lidar_assist_reason = safety.reason
            return replace(
                decision,
                active=True,
                steering=NEUTRAL_STEERING,
                throttle=NEUTRAL_THROTTLE,
                raw_steering=NEUTRAL_STEERING,
                raw_throttle=NEUTRAL_THROTTLE,
                action="lidar-stop",
                state="lidar-stop",
                reason=f"{decision.reason};lidar={safety.reason}",
            )
        if safety.status in {"slow", "caution"}:
            if decision.action not in {"continue", "speed-30", "speed-90", "cooldown", "confirming"}:
                self.lidar_assist_reason = f"action-{decision.action}:{safety.reason}"
                return decision
            correction = clamp(
                safety.steering_correction,
                -self.lidar_config.max_steering_correction,
                self.lidar_config.max_steering_correction,
                0.0,
            )
            steering = round(clamp(decision.steering + correction, -1.0, 1.0, NEUTRAL_STEERING), 3)
            raw_base = decision.raw_steering if decision.raw_steering is not None else decision.steering
            raw_steering = round(clamp(raw_base + correction, -1.0, 1.0, NEUTRAL_STEERING), 3)
            throttle = decision.throttle
            raw_throttle = decision.raw_throttle
            action = "lidar-caution"
            if safety.throttle_limit is not None:
                throttle = round(clamp(min(decision.throttle, safety.throttle_limit), 0.0, 1.0, NEUTRAL_THROTTLE), 3)
                if raw_throttle is not None:
                    raw_throttle = round(clamp(min(raw_throttle, safety.throttle_limit), 0.0, 1.0, NEUTRAL_THROTTLE), 3)
                action = "lidar-slow"
            self.lidar_assist_active = True
            self.lidar_assist_reason = f"{safety.reason}:{correction:+.3f}"
            return replace(
                decision,
                steering=steering,
                throttle=throttle,
                raw_steering=raw_steering,
                raw_throttle=raw_throttle,
                action=action,
                state="lidar-safety",
                reason=f"{decision.reason};lidar={safety.reason}:{correction:+.3f}",
            )

        self.lidar_assist_reason = safety.reason
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
            previous_mode = self.drive_mode
            requested_mode = normalize_drive_mode(mode)
            if previous_mode == "autonomous" and requested_mode == "manual":
                self._note_operator_event_locked(
                    "manual_override",
                    reason="operator-selected-manual",
                    details={"requested_mode": mode},
                )
            self.drive_mode = requested_mode
            if self.drive_mode == "autonomous":
                self.autonomous_controller.filter.reset()
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
            if self.drive_mode != "manual":
                steering_value = round(clamp(steering, -1.0, 1.0, NEUTRAL_STEERING), 3)
                throttle_value = round(clamp(throttle, -1.0, 1.0, NEUTRAL_THROTTLE), 3)
                is_neutral = (
                    abs(steering_value - NEUTRAL_STEERING) <= 0.01
                    and abs(throttle_value - NEUTRAL_THROTTLE) <= 0.01
                )
                if self.drive_mode == "autonomous" and not is_neutral:
                    self._note_operator_event_locked(
                        "manual_override_attempt",
                        reason="manual-control-post-during-autonomous",
                        details={
                            "source": source,
                            "requested_steering": steering_value,
                            "requested_throttle": throttle_value,
                        },
                    )
                if self.drive_mode == "autonomous":
                    self._apply_autonomous_control_locked()
                return self.control_snapshot_locked()

            self.drive_mode = "manual"
            if not ENABLE_WEB_CONTROL:
                self.control_armed = False
                self.control_source = "neutral"
                self.steering = NEUTRAL_STEERING
                self.throttle = NEUTRAL_THROTTLE
            else:
                steering_value = round(clamp(steering, -1.0, 1.0, NEUTRAL_STEERING), 3)
                throttle_value = round(clamp(throttle, -1.0, 1.0, NEUTRAL_THROTTLE), 3)
                is_neutral = (
                    abs(steering_value - NEUTRAL_STEERING) <= 0.01
                    and abs(throttle_value - NEUTRAL_THROTTLE) <= 0.01
                )
                self.control_armed = not is_neutral
                self.control_source = "neutral" if is_neutral else source
                self.steering = steering_value
                self.throttle = throttle_value
            self.control_updated_at = wall_time()
            self.control_seq += 1
            return self.control_snapshot_locked()

    def release_manual_control(self, source: str = "manual-release") -> dict[str, Any]:
        with self.lock:
            if self.drive_mode != "manual":
                if self.drive_mode == "autonomous":
                    self._apply_autonomous_control_locked()
                return self.control_snapshot_locked()

            self.control_armed = False
            self.control_source = source
            self.steering = NEUTRAL_STEERING
            self.throttle = NEUTRAL_THROTTLE
            self.control_updated_at = wall_time()
            self.control_seq += 1
            return self.control_snapshot_locked()

    def neutral(self, source: str = "neutral") -> dict[str, Any]:
        with self.lock:
            if self.drive_mode == "autonomous":
                self._note_operator_event_locked(
                    "manual_override",
                    reason=f"operator-{source}",
                    details={"source": source},
                )
            self.drive_mode = "manual"
            self.control_armed = False
            self.control_source = source
            self.steering = NEUTRAL_STEERING
            self.throttle = NEUTRAL_THROTTLE
            self.control_updated_at = wall_time()
            self.control_seq += 1
            return self.control_snapshot_locked()

    def set_steering_trim(self, value: Any) -> dict[str, Any]:
        trim = round(finite_float(value, name="steering_trim"), 3)
        with self.lock:
            self.steering_trim = trim
            self.control_updated_at = wall_time()
            self.control_seq += 1
            return self.control_snapshot_locked()

    def set_cruise_speed(self, value: Any) -> dict[str, Any]:
        speed = round(clamp(finite_float(value, name="cruise_speed"), 0.0, 1.0, AUTONOMOUS_CONFIG.cruise_throttle), 3)
        with self.lock:
            self.autonomous_config = replace(
                self.autonomous_config,
                crawl_throttle=speed,
                slow_throttle=speed,
                turn_throttle=speed,
                cruise_throttle=speed,
                fast_throttle=speed,
            )
            self.autonomous_controller.update_config(self.autonomous_config)
            self.autonomous_controller.speed_cap = speed
            if self.drive_mode == "autonomous":
                self._apply_autonomous_control_locked()
            else:
                self.control_updated_at = wall_time()
                self.control_seq += 1
            return {
                "mode": self.drive_mode,
                "control": self.control_snapshot_locked(),
                "autonomy": self.autonomy_snapshot_locked(now=wall_time()),
            }

    def set_turn_compensation(
        self,
        *,
        enabled: Any | None = None,
        interval_sec: Any | None = None,
        magnitude: Any | None = None,
        duration_sec: Any | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            enabled_value = self.turn_compensation_enabled if enabled is None else finite_bool(enabled, name="enabled")
            interval_value = (
                self.turn_compensation_interval_sec
                if interval_sec is None
                else round(max(0.0, finite_float(interval_sec, name="interval_sec")), 3)
            )
            magnitude_value = (
                self.turn_compensation_magnitude
                if magnitude is None
                else round(clamp(abs(finite_float(magnitude, name="magnitude")), 0.0, 2.0, 0.0), 3)
            )
            duration_value = (
                self.turn_compensation_duration_sec
                if duration_sec is None
                else round(max(0.0, finite_float(duration_sec, name="duration_sec")), 3)
            )
            if interval_value > 0.0:
                duration_value = min(duration_value, interval_value)

            changed = (
                enabled_value != self.turn_compensation_enabled
                or interval_value != self.turn_compensation_interval_sec
                or magnitude_value != self.turn_compensation_magnitude
                or duration_value != self.turn_compensation_duration_sec
            )
            self.turn_compensation_enabled = enabled_value
            self.turn_compensation_interval_sec = interval_value
            self.turn_compensation_magnitude = magnitude_value
            self.turn_compensation_duration_sec = duration_value
            if changed:
                now = wall_time()
                self.turn_compensation_last_pulse_at = now
                self.turn_compensation_active_until = 0.0
                self.turn_compensation_active = False
                self.turn_compensation_applied_correction = 0.0
                self.turn_compensation_reason = "waiting" if enabled_value else "disabled"
                self.control_updated_at = now
                self.control_seq += 1
            return self.turn_compensation_snapshot_locked(now=wall_time())

    def applied_steering_trim_locked(self) -> float:
        if (
            self.drive_mode == "autonomous"
            and self.autonomous_decision.action in {"turn-left", "turn-right"}
        ):
            return 0.0
        return self.steering_trim

    def control_snapshot_locked(self) -> dict[str, Any]:
        applied_trim = self.applied_steering_trim_locked()
        effective_steering = corrected_steering(self.steering, applied_trim)
        return {
            "armed": self.control_armed,
            "source": self.control_source,
            "mode": self.drive_mode,
            "steering": self.steering,
            "effective_steering": effective_steering,
            "steering_trim": self.steering_trim,
            "applied_steering_trim": applied_trim,
            "steering_trim_bypassed": applied_trim != self.steering_trim,
            "steering_trim_default": STEERING_TRIM,
            "throttle": self.throttle,
            "updated_age_sec": max(0.0, wall_time() - self.control_updated_at),
            "seq": self.control_seq,
        }

    def autonomous_config_snapshot_locked(self) -> dict[str, Any]:
        config = self.autonomous_config
        return {
            "min_confidence": config.min_confidence,
            "stale_prediction_sec": config.stale_prediction_sec,
            "max_frame_age_sec": config.max_frame_age_sec,
            "min_area_ratio": config.min_area_ratio,
            "near_area_ratio": config.near_area_ratio,
            "crawl_throttle": config.crawl_throttle,
            "slow_throttle": config.slow_throttle,
            "turn_throttle": config.turn_throttle,
            "cruise_throttle": config.cruise_throttle,
            "cruise_throttle_default": AUTONOMOUS_CONFIG.cruise_throttle,
            "fast_throttle": config.fast_throttle,
            "left_steering": config.left_steering,
            "right_steering": config.right_steering,
            "confirm_frames": config.confirm_frames,
            "safety_confirm_frames": config.safety_confirm_frames,
            "stop_hold_sec": config.stop_hold_sec,
            "turn_hold_sec": config.turn_hold_sec,
            "turn_pulse_enabled": config.turn_pulse_enabled,
            "turn_degrees": config.turn_degrees,
            "cooldown_sec": config.cooldown_sec,
            "dry_run": config.dry_run,
        }

    def turn_compensation_snapshot_locked(self, *, now: float | None = None) -> dict[str, Any]:
        now = wall_time() if now is None else now
        active = self.turn_compensation_active and now < self.turn_compensation_active_until
        next_in: float | None
        if not self.turn_compensation_enabled or self.turn_compensation_interval_sec <= 0.0:
            next_in = None
        elif active:
            next_in = 0.0
        elif self.turn_compensation_last_pulse_at is None:
            next_in = self.turn_compensation_interval_sec
        else:
            next_in = max(
                0.0,
                self.turn_compensation_interval_sec - (now - self.turn_compensation_last_pulse_at),
            )
        return {
            "enabled": self.turn_compensation_enabled,
            "interval_sec": self.turn_compensation_interval_sec,
            "duration_sec": self.turn_compensation_duration_sec,
            "magnitude": self.turn_compensation_magnitude,
            "direction": "right",
            "active": active,
            "applied_correction": self.turn_compensation_applied_correction if active else 0.0,
            "next_in_sec": rounded(next_in),
            "reason": self.turn_compensation_reason,
            "actions": sorted(TURN_COMPENSATION_ACTIONS),
        }

    def autonomy_snapshot_locked(self, *, now: float | None = None) -> dict[str, Any]:
        now = wall_time() if now is None else now
        return {
            "mode": self.drive_mode,
            "decision": self.autonomous_decision.to_status(),
            "config": self.autonomous_config_snapshot_locked(),
            "turn_compensation": self.turn_compensation_snapshot_locked(now=now),
        }

    def current_lane_config(self) -> LaneDetectorConfig:
        with self.lock:
            return self.lane_config

    def current_lane_guidance_locked(self, *, now: float | None = None) -> LaneGuidance | None:
        if self.lane_guidance is None:
            return None
        now = wall_time() if now is None else now
        age = 0.0 if self.lane_guidance_at is None else max(0.0, now - self.lane_guidance_at)
        return self.lane_guidance.with_age(age)

    def current_lane_guidance(self) -> LaneGuidance | None:
        with self.lock:
            return self.current_lane_guidance_locked(now=wall_time())

    def lane_snapshot_locked(self, *, now: float | None = None) -> dict[str, Any]:
        now = wall_time() if now is None else now
        guidance = self.current_lane_guidance_locked(now=now)
        lane_config = self.lane_config
        usable = False if guidance is None else guidance.is_usable(lane_config)
        if not lane_config.enabled:
            status = "disabled"
        elif self.lane_error:
            status = "error"
        elif self.lane_assist_active:
            status = "assisting"
        elif usable:
            status = "tracking"
        elif guidance is not None and guidance.detected:
            status = "weak"
        else:
            status = "searching"
        return {
            "enabled": lane_config.enabled,
            "status": status,
            "usable": usable,
            "assist_active": self.lane_assist_active,
            "applied_correction": round(self.lane_assist_correction, 3),
            "assist_reason": self.lane_assist_reason,
            "frames": self.lane_frames,
            "errors": self.lane_errors,
            "error": self.lane_error,
            "guidance": None if guidance is None else guidance.to_status(),
            "config": {
                "roi_top_ratio": lane_config.roi_top_ratio,
                "target_center_x": lane_config.target_center_x,
                "min_confidence": lane_config.min_confidence,
                "stale_sec": lane_config.stale_sec,
                "expected_lane_width_ratio": lane_config.expected_lane_width_ratio,
                "max_partial_lane_width_ratio": lane_config.max_partial_lane_width_ratio,
                "preferred_corridor": lane_config.preferred_corridor,
                "departure_center_error": lane_config.departure_center_error,
                "recovery_throttle": self.lane_recovery_throttle,
                "steering_gain": lane_config.steering_gain,
                "heading_gain": lane_config.heading_gain,
                "max_correction": lane_config.max_correction,
                "assist_actions": sorted(self.lane_assist_actions),
            },
        }

    def lidar_snapshot_locked(self, *, now: float | None = None) -> dict[str, Any]:
        now = wall_time() if now is None else now
        safety = analyze_lidar_scan(self.lidar_scan, config=self.lidar_config, now=now)
        self.lidar_safety = safety
        scan = self.lidar_scan
        return {
            "enabled": self.lidar_config.enabled,
            "status": safety.status if self.lidar_error is None else "error",
            "assist_active": self.lidar_assist_active,
            "assist_reason": self.lidar_assist_reason,
            "frames": self.lidar_frames,
            "errors": self.lidar_errors,
            "error": self.lidar_error,
            "source": None if scan is None else scan.source,
            "frame_id": None if scan is None else scan.frame_id,
            "received_age_sec": None if scan is None else rounded(scan.age(now)),
            "point_count": 0 if scan is None else len(scan.points),
            "points": lidar_status_points(scan, self.lidar_config),
            "safety": safety.to_status(),
            "config": {
                "stale_sec": self.lidar_config.stale_sec,
                "min_range_m": self.lidar_config.min_range_m,
                "max_range_m": self.lidar_config.max_range_m,
                "front_angle_deg": self.lidar_config.front_angle_deg,
                "stop_distance_m": self.lidar_config.stop_distance_m,
                "slow_distance_m": self.lidar_config.slow_distance_m,
                "caution_distance_m": self.lidar_config.caution_distance_m,
                "slow_throttle": self.lidar_config.slow_throttle,
                "max_steering_correction": self.lidar_config.max_steering_correction,
                "max_status_points": self.lidar_config.max_status_points,
            },
        }

    def _note_operator_event_locked(
        self,
        event_type: str,
        *,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.operator_event_seq += 1
        self.pending_operator_events.append(
            {
                "seq": self.operator_event_seq,
                "type": event_type,
                "reason": reason,
                "ts": round(wall_time(), 3),
                "mode": self.drive_mode,
                "control": {
                    "source": self.control_source,
                    "steering": self.steering,
                    "throttle": self.throttle,
                },
                "details": details or {},
            }
        )
        self.pending_operator_events = self.pending_operator_events[-16:]

    def _consume_operator_events_locked(self) -> list[dict[str, Any]]:
        events = list(self.pending_operator_events)
        self.pending_operator_events.clear()
        return events

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
                "lane": self.lane_snapshot_locked(now=now),
                "lidar": self.lidar_snapshot_locked(now=now),
                "control": self.control_snapshot_locked(),
                "autonomy": self.autonomy_snapshot_locked(now=now),
                "settings": self.settings_snapshot_locked(),
                "recording": self.recorder.snapshot(),
                "replayer": self.replayer.snapshot(),
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
        for key in ("class", "class_name", "confidence", "x", "y", "width", "height", "track_id", "track_hits"):
            value = prediction.get(key)
            if key in {"track_id", "track_hits"} and isinstance(value, (int, float)):
                item[key] = int(value)
            elif isinstance(value, (int, float)):
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
    if packet_type in {"I", "L"}:
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
    *,
    steering_trim: float | None = None,
) -> None:
    steering = corrected_steering(steering, steering_trim)
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
    lane = state_snapshot.get("lane", {})
    lane_guidance = lane.get("guidance") or {}
    lidar = state_snapshot.get("lidar", {})
    lidar_safety = lidar.get("safety") or {}
    lane_text = (
        "off"
        if not lane.get("enabled", False)
        else f"{lane.get('status', '-')}/{lane_guidance.get('correction', 0):+.2f}"
    )
    lidar_text = (
        "off"
        if not lidar.get("enabled", False)
        else f"{lidar.get('status', '-')}/{lidar_safety.get('min_front_distance_m', '-')}"
    )
    det = inf["detections"]
    latency = inf["latency_ms"]
    latency_text = "-" if latency is None else f"{latency}ms"
    if compact:
        lines = [
            f"f {context.seq}  det {det}  ia {inf['status']}",
            f"{control['mode']} {control['steering']:.2f}/{control['throttle']:.2f}  lidar {lidar_text}",
        ]
        scale = 0.42
        y = y0 + 26
    else:
        lines = [
            f"frame {context.seq}  det {det}  ia {inf['status']}  {latency_text}",
            f"rx {udp['packets']}  tx {udp['tx_packets']}  cliente {udp['last_client'] or '-'}",
            f"ctrl {control['mode']} {control['source']}  {control['steering']:.2f}/{control['throttle']:.2f}  auto {autonomy.get('action', '-')}  lane {lane_text}  lidar {lidar_text}",
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

    lane_guidance = state.current_lane_guidance()
    if lane_guidance is not None:
        frame = draw_lane_overlay(frame, lane_guidance, state.current_lane_config())

    frame = draw_status_overlay(frame, context, snapshot)
    return encode_jpeg(frame)


def build_placeholder(snapshot: dict[str, Any]) -> np.ndarray:
    width, height = 1280, 720
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (10, 10, 12)  # BGR for ~#0c0a0a — neutral near-black

    cx, cy = width // 2, height // 2
    cv2.circle(canvas, (cx, cy - 30), 26, (255, 166, 78), 2, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy - 30), 6, (255, 166, 78), -1, cv2.LINE_AA)

    title = "SIN SENAL"
    meta = f"UDP {snapshot['udp']['bind']}"

    (tw, _th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
    cv2.putText(
        canvas, title,
        (cx - tw // 2, cy + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (236, 236, 239), 2, cv2.LINE_AA,
    )
    (mw, _mh), _ = cv2.getTextSize(meta, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(
        canvas, meta,
        (cx - mw // 2, cy + 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (164, 164, 171), 1, cv2.LINE_AA,
    )
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

                started_ms = monotonic_ms()
                try:
                    payload = infer_one_frame(client, frame, config)
                    predictions = extract_predictions(payload)
                    latency = monotonic_ms() - started_ms
                    state.set_predictions(
                        seq,
                        predictions,
                        latency,
                        frame=frame,
                        inference_payload=payload,
                    )
                except Exception as exc:
                    state.set_inference_status("error", str(exc))
                    EXIT_EVENT.wait(INFERENCE_RETRY_SEC)
                    break

        except Exception as exc:
            state.set_inference_status("error", str(exc))
            EXIT_EVENT.wait(INFERENCE_RETRY_SEC)


def control_tx_loop(sock: socket.socket, state: RuntimeState) -> None:
    interval = 1.0 / CONTROL_TX_HZ
    while not EXIT_EVENT.wait(interval):
        address = state.get_client_address()
        if address is None:
            continue
        steering, throttle, control = state.get_control()
        try:
            send_control_packet(
                sock,
                address,
                steering,
                throttle,
                steering_trim=control.get(
                    "applied_steering_trim",
                    control.get("steering_trim"),
                ),
            )
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
        elif path == "/recording.json":
            self.send_json({"ok": True, "recording": self.state.recorder.snapshot()})
        elif path in {"/settings.json", "/control-settings.json"}:
            with self.state.lock:
                self.send_json({"ok": True, "settings": self.state.settings_snapshot_locked()})
        elif path == "/replayer.json":
            self.send_json(
                {
                    "ok": True,
                    "replayer": self.state.replayer.snapshot(
                        public_host=self.request_public_host()
                    ),
                }
            )
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
        if path in {"/recording", "/session-recording"}:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(min(length, 8192)) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_json({"ok": False, "error": "invalid json"}, status=400)
                return
            action = str(payload.get("action", "")).strip().lower()
            if action in {"start", "on", "enable"}:
                status = self.state.recorder.start()
            elif action in {"stop", "off", "disable"}:
                status = self.state.recorder.stop()
            elif "enabled" in payload:
                status = self.state.recorder.set_enabled(bool(payload.get("enabled")))
            else:
                status = self.state.recorder.set_enabled(not self.state.recorder.snapshot()["enabled"])
            self.send_json({"ok": status.get("last_error") is None, "recording": status})
            return
        if path in {"/replayer/start", "/retraining/start"}:
            status = self.state.replayer.start()
            status = self.state.replayer.snapshot(public_host=self.request_public_host())
            self.send_json({"ok": status.get("last_error") is None, "replayer": status})
            return
        if path in {"/replayer/stop", "/retraining/stop"}:
            status = self.state.replayer.stop()
            status = self.state.replayer.snapshot(public_host=self.request_public_host())
            self.send_json({"ok": status.get("last_error") is None, "replayer": status})
            return
        if path in {"/steering-trim", "/steering-compensation"}:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(min(length, 8192)) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_json({"ok": False, "error": "invalid json"}, status=400)
                return
            if "trim" in payload:
                trim_value = payload.get("trim")
            elif "steering_trim" in payload:
                trim_value = payload.get("steering_trim")
            elif "value" in payload:
                trim_value = payload.get("value")
            else:
                self.send_json({"ok": False, "error": "missing trim"}, status=400)
                return
            try:
                control = self.state.set_steering_trim(trim_value)
            except ValueError as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.send_json({"ok": True, "control": control})
            return
        if path in {"/cruise-speed", "/cruise-throttle"}:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(min(length, 8192)) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_json({"ok": False, "error": "invalid json"}, status=400)
                return
            if "speed" in payload:
                speed_value = payload.get("speed")
            elif "throttle" in payload:
                speed_value = payload.get("throttle")
            elif "value" in payload:
                speed_value = payload.get("value")
            else:
                self.send_json({"ok": False, "error": "missing speed"}, status=400)
                return
            try:
                status = self.state.set_cruise_speed(speed_value)
            except ValueError as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.send_json({"ok": True, **status})
            return
        if path in {"/turn-compensation", "/turn-compensation-pulse"}:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(min(length, 8192)) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self.send_json({"ok": False, "error": "invalid json"}, status=400)
                return
            interval_value = payload.get("interval_sec", payload.get("interval", payload.get("seconds")))
            duration_value = payload.get("duration_sec", payload.get("duration"))
            magnitude_value = payload.get("magnitude", payload.get("steering", payload.get("value")))
            try:
                status = self.state.set_turn_compensation(
                    enabled=payload.get("enabled") if "enabled" in payload else None,
                    interval_sec=interval_value,
                    magnitude=magnitude_value,
                    duration_sec=duration_value,
                )
            except ValueError as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.send_json({"ok": True, "turn_compensation": status})
            return
        if path in {"/settings", "/control-settings"}:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(min(length, 65536)) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
                settings = self.state.update_runtime_settings(payload)
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, status=400)
                return
            self.send_json({"ok": True, "settings": settings})
            return
        if path in {"/settings/defaults", "/control/defaults"}:
            settings = self.state.save_current_settings_as_defaults()
            self.send_json({"ok": settings.get("last_error") is None, "settings": settings})
            return
        if path in {"/control/neutral", "/neutral"}:
            self.send_json({"ok": True, "control": self.state.release_manual_control("neutral")})
            return
        if path in {"/control/stop", "/stop"}:
            self.send_json({"ok": True, "control": self.state.neutral("stop")})
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
        if action in {"stop", "estop"}:
            control = self.state.neutral("stop")
        elif action == "neutral":
            control = self.state.release_manual_control("neutral")
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

    def request_public_host(self) -> str:
        host_header = self.headers.get("Host", "").strip()
        if not host_header:
            return "127.0.0.1"
        if host_header.startswith("["):
            return host_header.split("]", 1)[0].strip("[]")
        return host_header.split(":", 1)[0]

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
            seq = state.update_frame(frame)
            state.record_frame_without_inference(seq, frame)
    elif packet_type == "L":
        state.update_lidar(payload)
    elif packet_type == "B":
        state.update_battery(payload)
    elif packet_type == "D":
        state.update_lidar_from_telemetry(payload)
        state.update_telemetry(payload)
    else:
        state.note_packet(packet_type, address, error="unknown packet type")

    steering, throttle, control = state.get_control()
    try:
        send_control_packet(
            sock,
            address,
            steering,
            throttle,
            steering_trim=control.get(
                "applied_steering_trim",
                control.get("steering_trim"),
            ),
        )
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
  <title>TP2 / Coche 4G</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: dark;
      --bg-0: #08080a;
      --bg-1: #111114;
      --bg-2: #17171b;
      --bg-3: #1f1f24;
      --line: #25252c;
      --line-soft: #1b1b21;
      --line-strong: #34343d;
      --ink: #ececef;
      --ink-2: #a4a4ab;
      --ink-3: #76767f;
      --muted: #54545c;
      --blue: #4ea6ff;
      --blue-soft: rgba(78,166,255,0.16);
      --blue-deep: #1a3a78;
      --cyan: #7dd3fc;
      --teal: #5eead4;
      --amber: #fbbf24;
      --red: #f87171;
      --green: #34d399;
      --display: "Space Grotesk", "IBM Plex Sans", -apple-system, BlinkMacSystemFont, sans-serif;
      --body: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      --shadow: 0 24px 50px rgba(0,0,0,0.55);
      --ring: 0 0 0 1px rgba(78,166,255,0.55), 0 0 0 4px rgba(78,166,255,0.10);
    }

    * { box-sizing: border-box; }

    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      background:
        radial-gradient(1200px 700px at 92% -10%, rgba(78,166,255,0.07), transparent 60%),
        radial-gradient(900px 600px at 5% 110%, rgba(94,234,212,0.04), transparent 55%),
        var(--bg-0);
      color: var(--ink);
      font-family: var(--body);
      font-size: 13.5px;
      letter-spacing: 0;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
      overflow: hidden;
    }

    body::before {
      content: "";
      position: fixed; inset: 0; pointer-events: none; z-index: 0;
      background-image:
        linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px);
      background-size: 32px 32px;
      mask-image: radial-gradient(circle at 50% 35%, black, transparent 75%);
    }

    .app {
      position: relative; z-index: 1;
      height: 100%;
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 14px;
      padding: 14px 18px 16px;
    }

    /* HEADER ----------------------------------------------------------------- */
    header {
      display: grid;
      grid-template-columns: minmax(260px, auto) 1fr auto;
      align-items: center;
      gap: 18px;
      padding-bottom: 11px;
      border-bottom: 1px solid var(--line);
    }

    .brand { display: flex; align-items: center; gap: 12px; }
    .brand .mark {
      width: 32px; height: 32px;
      border-radius: 8px;
      background: linear-gradient(135deg, var(--blue) 0%, var(--blue-deep) 100%);
      display: grid; place-items: center;
      box-shadow: 0 8px 22px rgba(78,166,255,0.32), inset 0 1px 0 rgba(255,255,255,0.18);
      position: relative;
    }
    .brand .mark::after {
      content: "";
      position: absolute; inset: -6px; border-radius: 14px;
      border: 1px solid rgba(78,166,255,0.18);
      pointer-events: none;
    }
    .brand .mark svg { width: 16px; height: 16px; color: #fff; }
    .brand-text { display: flex; flex-direction: column; gap: 2px; line-height: 1; }
    .brand h1 {
      margin: 0; font-family: var(--display);
      font-size: 17px; font-weight: 600; letter-spacing: -0.01em;
    }
    .brand h1 .accent { color: var(--blue); margin: 0 4px; font-weight: 400; }
    .brand h1 .sub { color: var(--ink-2); font-weight: 500; }
    .brand .meta {
      font-family: var(--mono);
      font-size: 10px; color: var(--ink-3);
      letter-spacing: 0.10em; text-transform: uppercase;
    }

    .pills { display: flex; gap: 7px; flex-wrap: wrap; justify-content: center; }
    .pill {
      display: inline-flex; align-items: center; gap: 8px;
      height: 28px; padding: 0 12px;
      border: 1px solid var(--line); border-radius: 999px;
      background: rgba(20,20,24,0.7);
      color: var(--ink-2);
      font-size: 10.5px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      font-weight: 600;
      white-space: nowrap;
    }
    .pill .label { color: var(--ink-3); font-weight: 500; }
    .pill .val { font-family: var(--mono); font-weight: 500; color: var(--ink); letter-spacing: 0; }
    .pill .dot { width: 6px; height: 6px; border-radius: 99px; background: var(--muted); box-shadow: 0 0 12px currentColor; }
    .pill.ok   { color: var(--cyan); }  .pill.ok   .dot { background: var(--cyan);  } .pill.ok   .val { color: var(--cyan); }
    .pill.warn { color: var(--amber); } .pill.warn .dot { background: var(--amber); } .pill.warn .val { color: var(--amber); }
    .pill.bad  { color: var(--red); }   .pill.bad  .dot { background: var(--red);   } .pill.bad  .val { color: var(--red); }

    .session { display: flex; align-items: center; gap: 18px; }
    .session .group { display: flex; flex-direction: column; align-items: flex-end; gap: 2px; }
    .session .label {
      color: var(--ink-3); font-size: 9px;
      letter-spacing: 0.18em; text-transform: uppercase;
      font-family: var(--mono);
    }
    .session .clock {
      font-family: var(--mono);
      font-size: 18px; color: var(--ink); font-weight: 500;
      line-height: 1.05;
      font-variant-numeric: tabular-nums;
    }
    .session .clock.accent { color: var(--blue); }

    /* MAIN GRID -------------------------------------------------------------- */
    main {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      gap: 16px;
    }

    /* LEFT COLUMN: video + deck --------------------------------------------- */
    .stage {
      min-height: 0;
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      gap: 10px;
    }

    .video {
      position: relative;
      border: 1px solid var(--line);
      border-radius: 14px;
      background:
        radial-gradient(120% 80% at 50% 0%, rgba(78,166,255,0.05) 0%, transparent 60%),
        radial-gradient(120% 80% at 50% 100%, rgba(0,0,0,0.4) 0%, transparent 60%),
        #0b0b0f;
      overflow: hidden;
      box-shadow: var(--shadow);
      min-height: 0;
    }
    .video img {
      position: absolute; inset: 0;
      width: 100%; height: 100%;
      object-fit: contain;
      display: block;
      z-index: 1;
      transition: opacity 200ms ease;
    }
    .video.no-feed img { opacity: 0; }
    .lidar-canvas {
      position: absolute; inset: 0;
      width: 100%; height: 100%;
      z-index: 1;
      opacity: 0;
      background:
        linear-gradient(180deg, rgba(8,8,12,0.22), rgba(8,8,12,0.72)),
        #06070a;
      transition: opacity 160ms ease;
    }
    .video.lidar-mode img { opacity: 0; }
    .video.lidar-mode .lidar-canvas { opacity: 1; }
    .view-toggle {
      position: absolute; left: 18px; top: 18px; z-index: 4;
      display: inline-grid; grid-template-columns: 1fr 1fr; gap: 3px;
      padding: 3px;
      border: 1px solid rgba(78,166,255,0.24);
      border-radius: 8px;
      background: rgba(8,8,12,0.78);
      backdrop-filter: blur(8px);
    }
    .view-toggle button {
      height: 28px; min-width: 84px; padding: 0 10px;
      border: 0; border-radius: 5px;
      background: transparent;
      color: var(--ink-2);
      cursor: pointer;
      font-family: var(--display); font-weight: 600; font-size: 10px;
      letter-spacing: 0.11em; text-transform: uppercase;
      display: inline-flex; align-items: center; justify-content: center; gap: 6px;
    }
    .view-toggle button.active { background: var(--blue); color: #061226; box-shadow: 0 4px 18px rgba(78,166,255,0.24); }
    .view-toggle button svg { width: 12px; height: 12px; }
    .video::after {
      content: "";
      position: absolute; inset: 14px;
      border-radius: 8px;
      pointer-events: none;
      z-index: 3;
      background:
        linear-gradient(to right, var(--blue) 0 14px, transparent 14px) top left/14px 1px no-repeat,
        linear-gradient(to bottom, var(--blue) 0 14px, transparent 14px) top left/1px 14px no-repeat,
        linear-gradient(to left, var(--blue) 0 14px, transparent 14px) top right/14px 1px no-repeat,
        linear-gradient(to bottom, var(--blue) 0 14px, transparent 14px) top right/1px 14px no-repeat,
        linear-gradient(to right, var(--blue) 0 14px, transparent 14px) bottom left/14px 1px no-repeat,
        linear-gradient(to top, var(--blue) 0 14px, transparent 14px) bottom left/1px 14px no-repeat,
        linear-gradient(to left, var(--blue) 0 14px, transparent 14px) bottom right/14px 1px no-repeat,
        linear-gradient(to top, var(--blue) 0 14px, transparent 14px) bottom right/1px 14px no-repeat;
      opacity: 0.20;
    }

    .no-feed-overlay { position: absolute; inset: 14px; display: none; place-items: center; pointer-events: none; z-index: 2; }
    .video.no-feed .no-feed-overlay { display: grid; }
    .no-feed-card { display: flex; flex-direction: column; align-items: center; gap: 14px; padding: 20px 28px; max-width: 80%; text-align: center; }
    .no-feed-card .pulse {
      width: 36px; height: 36px;
      border-radius: 50%;
      border: 1.5px solid var(--blue);
      position: relative; display: grid; place-items: center;
    }
    .no-feed-card .pulse::before, .no-feed-card .pulse::after {
      content: ""; position: absolute; inset: -1.5px; border-radius: 50%;
      border: 1.5px solid var(--blue);
      opacity: 0;
      animation: pulse-ring 2.4s cubic-bezier(0.2,0.6,0.3,1) infinite;
    }
    .no-feed-card .pulse::after { animation-delay: 1.2s; }
    .no-feed-card .pulse .core { width: 8px; height: 8px; border-radius: 50%; background: var(--blue); box-shadow: 0 0 12px var(--blue); }
    @keyframes pulse-ring { 0%{transform:scale(0.85);opacity:.55} 80%{transform:scale(2.2);opacity:0} 100%{transform:scale(2.2);opacity:0} }
    .no-feed-title { font-family: var(--body); font-weight: 500; font-size: 14px; letter-spacing: 0.22em; text-transform: uppercase; color: var(--ink); margin: 0; }
    .no-feed-meta { font-family: var(--mono); font-size: 11px; color: var(--ink-3); letter-spacing: 0.04em; margin: 0; }

    .rec {
      position: absolute; right: 18px; top: 18px;
      display: flex; align-items: center; gap: 8px;
      padding: 6px 10px;
      background: rgba(8,8,12,0.78); backdrop-filter: blur(8px);
      border: 1px solid rgba(78,166,255,0.30);
      border-radius: 4px;
      font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.16em;
      color: var(--blue); z-index: 4;
    }
    .rec .blink { width: 7px; height: 7px; border-radius: 99px; background: var(--blue); box-shadow: 0 0 12px var(--blue); }
    .rec.active { border-color: rgba(248,113,113,0.5); color: var(--red); }
    .rec.active .blink { background: var(--red); box-shadow: 0 0 14px var(--red); animation: blink 1.4s ease-in-out infinite; }
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

    .hud { position: absolute; left: 18px; bottom: 18px; display: flex; gap: 8px; flex-wrap: wrap; pointer-events: none; z-index: 4; }
    .hud .chip {
      background: rgba(8,8,12,0.78); backdrop-filter: blur(8px);
      border: 1px solid rgba(78,166,255,0.22);
      border-radius: 4px;
      padding: 6px 10px;
      font-family: var(--mono); font-size: 11px;
      color: var(--ink);
      display: inline-flex; align-items: baseline; gap: 6px;
      font-variant-numeric: tabular-nums;
    }
    .hud .chip span {
      color: var(--blue); text-transform: uppercase;
      font-size: 9px; letter-spacing: 0.16em;
      font-family: var(--body); font-weight: 500;
    }
    .hud .chip strong { font-weight: 500; }

    /* DECK -------------------------------------------------------------------- */
    .deck {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
      grid-template-rows: 1fr auto;
      column-gap: 18px;
      row-gap: 10px;
      align-items: center;
      padding: 12px 16px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(26,26,30,0.78), rgba(17,17,21,0.80));
      border-radius: 14px;
    }

    .deck .group { display: flex; flex-direction: column; gap: 8px; min-width: 0; }
    .deck h3 { margin: 0; font-family: var(--mono); font-size: 9.5px; color: var(--ink-3); letter-spacing: 0.18em; text-transform: uppercase; font-weight: 600; }

    .steer-wrap { display: flex; align-items: center; gap: 14px; min-width: 0; }
    .steer-meter {
      flex: 1 1 auto;
      min-width: 110px; max-width: 220px; height: 30px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(78,166,255,0.06), transparent 50%, rgba(78,166,255,0.06));
      position: relative; overflow: hidden;
    }
    .steer-meter .mid { position: absolute; top: -2px; bottom: -2px; left: 50%; width: 1px; background: var(--line); box-shadow: 0 0 0 0.5px rgba(78,166,255,0.20); }
    .steer-meter .tick { position: absolute; top: 0; bottom: 0; width: 1px; background: rgba(78,166,255,0.12); }
    .steer-meter .fill-left { position: absolute; top: 4px; bottom: 4px; right: 50%; width: 0%; background: linear-gradient(270deg, var(--blue), #a8d0ff); border-radius: 4px 0 0 4px; transition: width 90ms ease; }
    .steer-meter .fill-right { position: absolute; top: 4px; bottom: 4px; left: 50%; width: 0%; background: linear-gradient(90deg, var(--blue), #a8d0ff); border-radius: 0 4px 4px 0; transition: width 90ms ease; }
    .axis-data { font-family: var(--mono); display: flex; flex-direction: column; gap: 2px; min-width: 70px; }
    .axis-data .v { font-size: 22px; color: var(--blue); font-weight: 500; line-height: 1; font-variant-numeric: tabular-nums; }
    .axis-data .l { font-size: 9.5px; color: var(--ink-3); letter-spacing: 0.18em; text-transform: uppercase; font-family: var(--mono); font-weight: 600; }
    .axis-data .l.dir { color: var(--ink-2); letter-spacing: 0; text-transform: none; font-size: 11px; font-weight: 500; }

    .keys-wrap { display: flex; flex-direction: column; align-items: center; gap: 7px; align-self: center; justify-self: center; }
    .keys { display: grid; grid-template-columns: repeat(3, 36px); grid-template-rows: 36px 36px; gap: 4px; }
    .key {
      border: 1px solid var(--line);
      background: var(--bg-2); border-radius: 6px;
      display: grid; place-items: center;
      font-family: var(--mono); font-weight: 500; font-size: 12px;
      color: var(--ink-2);
      transition: all 80ms ease;
    }
    .key.empty { border-color: transparent; background: transparent; }
    .key.k-w { grid-column: 2; grid-row: 1; }
    .key.k-a { grid-column: 1; grid-row: 2; }
    .key.k-s { grid-column: 2; grid-row: 2; }
    .key.k-d { grid-column: 3; grid-row: 2; }
    .key.active { background: var(--blue); color: #061226; border-color: var(--blue); box-shadow: 0 0 18px rgba(78,166,255,0.5); transform: translateY(1px); font-weight: 600; }
    .key.brake { background: linear-gradient(180deg, #3a1820, #240e14); color: var(--red); border-color: rgba(248,113,113,0.45); box-shadow: 0 0 18px rgba(248,113,113,0.32); }
    .keys-caption { font-family: var(--mono); font-size: 9.5px; color: var(--ink-3); letter-spacing: 0.10em; text-transform: uppercase; font-weight: 600; }
    .keys-caption .kbd { display: inline-block; padding: 1px 6px; border: 1px solid var(--line); border-radius: 3px; color: var(--ink-2); margin: 0 1px; font-family: var(--mono); }

    .throttle-wrap { display: flex; align-items: center; gap: 14px; flex-direction: row-reverse; min-width: 0; justify-content: flex-end; }
    .throttle-meter {
      width: 28px; height: 96px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(248,113,113,0.06), transparent 50%, rgba(78,166,255,0.06));
      position: relative; overflow: hidden;
    }
    .throttle-meter .mid { position: absolute; left: -2px; right: -2px; top: 50%; height: 1px; background: var(--line); box-shadow: 0 0 0 0.5px rgba(78,166,255,0.20); }
    .throttle-meter .tick { position: absolute; left: 0; right: 0; height: 1px; background: rgba(78,166,255,0.12); }
    .throttle-meter .fill-fwd { position: absolute; left: 4px; right: 4px; bottom: 50%; height: 0%; background: linear-gradient(0deg, var(--blue), #a8d0ff); border-radius: 4px 4px 0 0; transition: height 90ms ease; }
    .throttle-meter .fill-rev { position: absolute; left: 4px; right: 4px; top: 50%; height: 0%; background: linear-gradient(180deg, var(--red), #fca5a5); border-radius: 0 0 4px 4px; transition: height 90ms ease; }
    .throttle-wrap .axis-data { align-items: flex-start; }

    .deck-actions {
      grid-column: 1 / -1;
      display: flex; gap: 10px; align-items: center;
      padding-top: 10px;
      border-top: 1px solid var(--line-soft);
    }
    .mode-toggle {
      display: inline-grid;
      grid-template-columns: 1fr 1fr;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 3px;
      background: var(--bg-1);
      gap: 3px;
    }
    .mode-toggle button {
      height: 32px; min-width: 100px; padding: 0 14px;
      border: 0; border-radius: 5px;
      background: transparent;
      color: var(--ink-2);
      cursor: pointer;
      font-family: var(--display); font-weight: 600; font-size: 11px;
      letter-spacing: 0.10em; text-transform: uppercase;
      transition: all 100ms ease;
      display: inline-flex; align-items: center; justify-content: center; gap: 6px;
    }
    .mode-toggle button:hover { color: var(--ink); }
    .mode-toggle button.active { background: var(--blue); color: #061226; box-shadow: 0 4px 18px rgba(78,166,255,0.28), inset 0 1px 0 rgba(255,255,255,0.16); }
    .mode-toggle button svg { width: 12px; height: 12px; }
    .deck-actions .grow { flex: 1 1 auto; }

    button.action {
      height: 36px; padding: 0 14px;
      border: 1px solid var(--line);
      background: var(--bg-2);
      color: var(--ink-2);
      font-family: var(--body); font-size: 11px; font-weight: 600;
      letter-spacing: 0.10em; text-transform: uppercase;
      border-radius: 7px;
      cursor: pointer;
      display: inline-flex; align-items: center; gap: 7px;
      transition: all 120ms ease;
    }
    button.action:hover { color: var(--ink); border-color: var(--line-strong); background: var(--bg-3); }
    button.action svg { width: 13px; height: 13px; }

    button.action.btn-record {
      border-color: rgba(78,166,255,0.40);
      background: rgba(78,166,255,0.08);
      color: var(--blue);
    }
    button.action.btn-record:hover { background: rgba(78,166,255,0.16); }
    button.action.btn-record.active { background: rgba(248,113,113,0.16); color: var(--red); border-color: rgba(248,113,113,0.55); box-shadow: 0 0 18px rgba(248,113,113,0.18); }

    button.action.btn-review { border-color: rgba(125,211,252,0.40); background: rgba(125,211,252,0.08); color: var(--cyan); }
    button.action.btn-review:hover { background: rgba(125,211,252,0.16); }
    button.action.btn-review.active { background: rgba(125,211,252,0.18); box-shadow: 0 0 18px rgba(125,211,252,0.18); }

    button.stop {
      height: 36px; padding: 0 22px;
      border: 1px solid rgba(248,113,113,0.5);
      background: linear-gradient(180deg, #3a1820, #240e14);
      color: var(--red);
      font-family: var(--display); font-size: 12px; font-weight: 700;
      letter-spacing: 0.20em;
      border-radius: 7px;
      cursor: pointer;
      text-transform: uppercase;
      transition: all 120ms ease;
      display: inline-flex; align-items: center; gap: 8px;
    }
    button.stop:hover { background: linear-gradient(180deg, #4a1d28, #2e1218); box-shadow: 0 0 24px rgba(248,113,113,0.30); }
    button.stop:active { transform: translateY(1px); }
    button.stop svg { width: 12px; height: 12px; }

    /* RIGHT COLUMN ----------------------------------------------------------- */
    .side {
      min-height: 0;
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 10px;
      padding-right: 2px;
    }

    /* CONTEXT STRIP ---------------------------------------------------------- */
    .ctx-strip {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1px;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: var(--line);
    }
    .ctx-strip .ctx {
      display: flex; flex-direction: column; gap: 2px;
      padding: 9px 12px;
      background: linear-gradient(180deg, rgba(26,26,30,0.80), rgba(17,17,21,0.80));
      min-width: 0;
    }
    .ctx-strip .ctx .label { font-family: var(--mono); font-size: 9px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-3); }
    .ctx-strip .ctx .value { font-family: var(--mono); font-size: 13px; color: var(--ink); font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .ctx-strip .ctx .value.tag { display: inline-flex; align-items: center; gap: 6px; font-size: 11.5px; font-weight: 500; }
    .ctx-strip .ctx .value .dot { width: 6px; height: 6px; border-radius: 99px; background: currentColor; box-shadow: 0 0 8px currentColor; }
    .ctx-strip .ctx.action-continue .value { color: var(--teal); }
    .ctx-strip .ctx.action-stop .value { color: var(--red); }
    .ctx-strip .ctx.action-turn .value { color: var(--amber); }
    .ctx-strip .ctx.action-yield .value { color: var(--cyan); }
    .ctx-strip .ctx.action-default .value { color: var(--ink); }

    /* TABS ------------------------------------------------------------------- */
    .tabs {
      display: grid; grid-template-columns: repeat(4, 1fr);
      gap: 3px; padding: 3px;
      border-radius: 10px; border: 1px solid var(--line);
      background: var(--bg-2);
    }
    .tab {
      height: 30px; padding: 0 8px;
      border: 0; background: transparent;
      color: var(--ink-3);
      font-family: var(--display); font-size: 10.5px; font-weight: 600;
      letter-spacing: 0.14em; text-transform: uppercase;
      border-radius: 7px; cursor: pointer;
      display: flex; align-items: center; justify-content: center; gap: 6px;
      white-space: nowrap;
    }
    .tab .badge {
      font-family: var(--mono); font-size: 9px; font-weight: 500;
      color: var(--ink-3);
      padding: 1px 5px; border-radius: 99px;
      background: var(--bg-1); border: 1px solid var(--line);
    }
    .tab:hover { color: var(--ink-2); }
    .tab.active { background: linear-gradient(180deg, rgba(26,26,30,0.95), rgba(17,17,21,0.95)); color: var(--ink); box-shadow: 0 0 0 1px var(--line-strong), inset 0 1px 0 rgba(255,255,255,0.04); }
    .tab.active .badge { color: var(--blue); border-color: rgba(78,166,255,0.45); }
    .tab.active .badge.warn { color: var(--amber); border-color: rgba(251,191,36,0.45); }
    .tab.active .badge.bad { color: var(--red); border-color: rgba(248,113,113,0.45); }
    .tab.active .badge.ok { color: var(--teal); border-color: rgba(94,234,212,0.45); }

    .tab-host {
      min-height: 0; overflow-y: auto;
      display: grid; gap: 12px; align-content: start;
      padding-right: 4px;
      scrollbar-width: thin; scrollbar-color: var(--line-strong) transparent;
    }
    .tab-host::-webkit-scrollbar { width: 8px; }
    .tab-host::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 99px; }

    .tab-panel { display: none; gap: 12px; grid-template-columns: 1fr; }
    .tab-panel.active { display: grid; animation: tabIn .25s ease both; }
    @keyframes tabIn { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: none; } }

    /* CARDS ------------------------------------------------------------------ */
    .card {
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(26,26,30,0.78), rgba(17,17,21,0.78));
      border-radius: 12px;
      padding: 13px 14px 12px;
    }
    .card h2 {
      display: flex; align-items: center; gap: 8px;
      margin: 0 0 10px;
      font-family: var(--mono);
      font-weight: 600; font-size: 10.5px; letter-spacing: 0.18em; text-transform: uppercase;
      color: var(--ink-2);
    }
    .card h2 .glyph { width: 14px; height: 14px; display: grid; place-items: center; color: var(--blue); }
    .card h2 .tag {
      margin-left: auto;
      font-family: var(--mono); font-weight: 500;
      font-size: 10px; letter-spacing: 0.06em;
      color: var(--ink-3);
      padding: 2px 8px; border-radius: 999px; border: 1px solid var(--line);
      max-width: 180px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .card h3 {
      margin: 12px 0 8px;
      font-family: var(--mono); font-weight: 600;
      font-size: 9.5px; color: var(--ink-3);
      letter-spacing: 0.18em; text-transform: uppercase;
    }

    .row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: baseline; padding: 6px 0; border-bottom: 1px solid var(--line-soft); }
    .row:last-child { border-bottom: 0; }
    .row .k { color: var(--ink-3); font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; font-size: 10.5px; }
    .row .v { font-family: var(--mono); color: var(--ink); text-align: right; font-size: 12px; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
    .row .v.accent { color: var(--blue); }
    .row .v.cyan { color: var(--cyan); }
    .row .v.amber { color: var(--amber); }
    .row .v.red { color: var(--red); }
    .row .v.muted { color: var(--ink-3); }

    /* TUNING CONTROLS -------------------------------------------------------- */
    .control-block { display: grid; gap: 10px; }
    .control-block .header-line {
      display: flex; align-items: baseline; justify-content: space-between; gap: 10px;
    }
    .control-block .header-line .title {
      font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.16em; text-transform: uppercase;
      color: var(--ink-2); font-weight: 600;
    }
    .control-block .header-line .tag {
      font-family: var(--mono); font-size: 10px; color: var(--ink-3);
      padding: 2px 7px; border-radius: 99px; border: 1px solid var(--line);
    }

    .readout-line {
      display: grid; grid-template-columns: auto 1fr auto; gap: 12px; align-items: baseline;
      padding: 6px 0;
    }
    .readout-line .value { font-family: var(--mono); font-size: 26px; line-height: 1; color: var(--blue); font-weight: 500; font-variant-numeric: tabular-nums; }
    .readout-line .dir { font-family: var(--mono); color: var(--ink-2); font-size: 11.5px; text-align: right; }

    .slider-row { display: grid; grid-template-columns: auto 1fr auto; gap: 10px; align-items: center; color: var(--ink-3); font-family: var(--mono); font-size: 10px; }
    .slider-row input[type="range"] {
      -webkit-appearance: none; appearance: none;
      width: 100%; height: 6px; padding: 0;
      background: linear-gradient(90deg, var(--blue) 0%, var(--blue) var(--seek-pct,50%), var(--bg-3) var(--seek-pct,50%));
      border: 1px solid var(--line);
      border-radius: 99px;
    }
    .slider-row input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 14px; height: 14px; border-radius: 99px;
      background: var(--ink);
      border: 2px solid var(--blue);
      box-shadow: 0 0 0 4px rgba(78,166,255,0.18);
      cursor: ew-resize;
    }
    .slider-row input[type="range"]::-moz-range-thumb {
      width: 14px; height: 14px; border-radius: 99px;
      background: var(--ink); border: 2px solid var(--blue);
      box-shadow: 0 0 0 4px rgba(78,166,255,0.18); cursor: ew-resize;
    }

    .input-row { display: grid; grid-template-columns: 1fr auto; gap: 8px; }
    .input-row input {
      width: 100%; min-width: 0; height: 34px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--bg-1); color: var(--ink);
      padding: 0 10px;
      font-family: var(--mono); font-size: 12px;
      font-variant-numeric: tabular-nums;
      outline: none;
      transition: border-color .12s, box-shadow .12s;
    }
    .input-row input:focus { border-color: rgba(78,166,255,0.75); box-shadow: var(--ring); }
    .input-row button {
      height: 34px; padding: 0 12px;
      border: 1px solid var(--line);
      background: var(--bg-2);
      color: var(--ink-2);
      font-family: var(--body); font-size: 11px; font-weight: 600;
      letter-spacing: 0.10em; text-transform: uppercase;
      border-radius: 7px; cursor: pointer;
    }
    .input-row button:hover { color: var(--ink); border-color: var(--line-strong); background: var(--bg-3); }

    .toggle-line {
      display: flex; align-items: center; justify-content: space-between;
      gap: 10px; padding: 8px 10px;
      border: 1px solid var(--line); border-radius: 7px;
      background: var(--bg-2);
    }
    .toggle-line .lbl { font-size: 12px; color: var(--ink-2); }
    .toggle-line input[type="checkbox"] { width: auto; padding: 0; accent-color: var(--blue); }

    .compact-fields { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .compact-field {
      min-width: 0; display: grid; gap: 4px;
      color: var(--ink-3); font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.10em; text-transform: uppercase; font-weight: 600;
    }
    .compact-field input {
      min-width: 0; height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--bg-1); color: var(--ink);
      padding: 0 8px;
      font-family: var(--mono); font-size: 12px;
      font-variant-numeric: tabular-nums;
      outline: none;
    }
    .compact-field input:focus { border-color: rgba(78,166,255,0.75); box-shadow: var(--ring); }

    .save-defaults-row {
      display: grid; gap: 8px;
      margin-top: 4px; padding: 10px 12px;
      border: 1px dashed rgba(251,191,36,0.32);
      border-radius: 8px;
      background: rgba(251,191,36,0.04);
    }
    .save-defaults-row button {
      height: 36px;
      border: 1px solid rgba(251,191,36,0.45);
      background: rgba(251,191,36,0.10);
      color: var(--amber);
      font-family: var(--body); font-size: 11px; font-weight: 600;
      letter-spacing: 0.12em; text-transform: uppercase;
      border-radius: 7px; cursor: pointer;
    }
    .save-defaults-row button:hover { background: rgba(251,191,36,0.18); }
    .save-defaults-row .path { font-family: var(--mono); font-size: 10px; color: var(--ink-3); overflow-wrap: anywhere; }

    /* SPARKLINES ------------------------------------------------------------- */
    .spark-wrap { display: grid; grid-template-columns: 1fr auto; align-items: end; gap: 14px; padding: 6px 0 2px; }
    .spark { height: 36px; width: 100%; }
    .spark path { fill: none; stroke: var(--blue); stroke-width: 1.4; stroke-linejoin: round; stroke-linecap: round; }
    .spark .area { fill: rgba(78,166,255,0.14); stroke: none; }
    .spark .grid { stroke: var(--line-soft); stroke-width: 1; stroke-dasharray: 2 4; }
    .spark.cyan path { stroke: var(--cyan); }
    .spark.cyan .area { fill: rgba(125,211,252,0.13); }
    .spark-data { font-family: var(--mono); text-align: right; min-width: 80px; }
    .spark-data .v { font-size: 18px; color: var(--ink); font-weight: 500; line-height: 1; font-variant-numeric: tabular-nums; }
    .spark-data .v small { font-size: 0.6em; color: var(--ink-3); font-weight: 400; margin-left: 3px; }
    .spark-data .l { font-size: 9.5px; color: var(--ink-3); letter-spacing: 0.14em; text-transform: uppercase; margin-top: 4px; font-family: var(--mono); font-weight: 600; }

    /* DETECTIONS ------------------------------------------------------------- */
    .detections { display: grid; gap: 6px; max-height: 220px; overflow: auto; padding-right: 2px; }
    .detections::-webkit-scrollbar { width: 4px; }
    .detections::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 99px; }
    .det {
      display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 12px;
      padding: 7px 10px;
      border: 1px solid var(--line); border-radius: 7px;
      background: rgba(20,20,24,0.55);
    }
    .det .name { font-weight: 500; font-size: 12px; color: var(--ink); }
    .det .conf { display: flex; align-items: center; gap: 8px; font-family: var(--mono); font-size: 11px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
    .det .conf .meter { width: 50px; height: 4px; background: var(--bg-3); border-radius: 2px; overflow: hidden; border: 1px solid var(--line); }
    .det .conf .meter .fill { height: 100%; background: linear-gradient(90deg, var(--blue), var(--cyan)); width: 0%; transition: width 240ms ease; }
    .det.empty { text-align: center; color: var(--ink-3); font-size: 12px; border-style: dashed; grid-template-columns: 1fr; padding: 14px; }

    .empty-state {
      padding: 14px 12px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--ink-3);
      text-align: center;
      font-size: 11.5px;
      font-family: var(--mono);
    }

    /* RESPONSIVE ------------------------------------------------------------- */
    @media (max-width: 1080px) {
      html, body { overflow: auto; }
      .app { height: auto; min-height: 100%; }
      header { grid-template-columns: 1fr; gap: 14px; }
      .pills { justify-content: flex-start; }
      .session { justify-content: flex-start; gap: 18px; }
      .session .group { align-items: flex-start; }
      main { grid-template-columns: 1fr; }
      .video { aspect-ratio: 16 / 9; }
      .side { grid-template-rows: auto auto auto; max-height: none; overflow: visible; }
    }
    @media (max-width: 720px) {
      .deck { grid-template-columns: 1fr; row-gap: 14px; }
      .deck .group { align-items: center; justify-self: center; }
      .deck-actions { flex-direction: column; align-items: stretch; }
      .mode-toggle button { min-width: 0; }
      .brand h1 { font-size: 16px; }
      .tabs { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <div class="mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="9"/>
            <path d="M12 3v9l5.5 3.2"/>
          </svg>
        </div>
        <div class="brand-text">
          <h1>TP2<span class="accent">/</span><span class="sub">Coche 4G</span></h1>
          <span class="meta">EPC · Roboflow · UDP 20001</span>
        </div>
      </div>

      <div class="pills">
        <div class="pill warn" id="pill-link"><span class="dot"></span><span class="label">4G</span><span class="val" id="pill-link-val">--</span></div>
        <div class="pill warn" id="pill-video"><span class="dot"></span><span class="label">Video</span><span class="val" id="pill-video-val">--</span></div>
        <div class="pill warn" id="pill-ai"><span class="dot"></span><span class="label">IA</span><span class="val" id="pill-ai-val">--</span></div>
        <div class="pill warn" id="pill-lane"><span class="dot"></span><span class="label">Carril</span><span class="val" id="pill-lane-val">--</span></div>
        <div class="pill warn" id="pill-lidar"><span class="dot"></span><span class="label">LiDAR</span><span class="val" id="pill-lidar-val">--</span></div>
        <div class="pill bad" id="pill-control"><span class="dot"></span><span class="label">Control</span><span class="val" id="pill-control-val">OFF</span></div>
        <div class="pill warn" id="pill-recording"><span class="dot"></span><span class="label">Dataset</span><span class="val" id="pill-recording-val">OFF</span></div>
      </div>

      <div class="session">
        <div class="group">
          <span class="label">Sesion</span>
          <span class="clock accent" id="session-clock">00:00:00</span>
        </div>
        <div class="group">
          <span class="label">Hora</span>
          <span class="clock" id="wall-clock">--:--:--</span>
        </div>
      </div>
    </header>

    <main>
      <section class="stage">
        <div class="video camera-mode" id="video-shell">
          <img id="video" src="/video.mjpg" alt="Camara del coche">
          <canvas class="lidar-canvas" id="lidar-canvas" aria-label="Reconstruccion 3D del LiDAR"></canvas>
          <div class="view-toggle" role="group" aria-label="Vista principal">
            <button type="button" id="view-camera" class="active" title="Ver camara">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/></svg>
              Camara
            </button>
            <button type="button" id="view-lidar" title="Ver reconstruccion LiDAR">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v18"/><path d="M4 7l8-4 8 4"/><path d="M4 17l8 4 8-4"/><path d="M4 7v10l8 4 8-4V7"/></svg>
              LiDAR
            </button>
          </div>
          <div class="no-feed-overlay" aria-hidden="true">
            <div class="no-feed-card">
              <div class="pulse"><span class="core"></span></div>
              <p class="no-feed-title">Sin senal</p>
              <p class="no-feed-meta" id="no-feed-meta">Esperando cuadro de camara...</p>
            </div>
          </div>
          <div class="rec" id="rec-badge"><span class="blink"></span><span id="rec-badge-text">EN VIVO</span></div>
          <div class="hud">
            <div class="chip"><span>FPS</span><strong id="hud-fps">--</strong></div>
            <div class="chip"><span>Lat</span><strong id="hud-lat">-- ms</strong></div>
            <div class="chip"><span>Det</span><strong id="hud-det">--</strong></div>
            <div class="chip"><span>Frame</span><strong id="hud-frame">--</strong></div>
            <div class="chip"><span>Obs</span><strong id="hud-lidar">--</strong></div>
          </div>
        </div>

        <div class="deck">
          <div class="group steer-wrap">
            <div class="axis-data">
              <span class="l">Giro</span>
              <span class="v" id="steer-val">0.25</span>
              <span class="l dir" id="steer-dir">centrado</span>
            </div>
            <div class="steer-meter" aria-label="Giro">
              <div class="tick" style="left:25%"></div>
              <div class="mid"></div>
              <div class="tick" style="left:75%"></div>
              <div class="fill-left" id="steer-left"></div>
              <div class="fill-right" id="steer-right"></div>
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
            <div class="keys-caption">freno <span class="kbd">SP</span> <span class="kbd">X</span></div>
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
            <div class="mode-toggle" role="group" aria-label="Modo de conduccion">
              <button type="button" id="mode-manual" class="active" title="Conduccion manual (M)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 21V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16"/><path d="M5 21h14"/><path d="M9 9h6"/><path d="M9 13h6"/></svg>
                Manual
              </button>
              <button type="button" id="mode-auto" title="Conduccion autonoma (N)">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"/><path d="M12 18v4"/><circle cx="12" cy="12" r="6"/><path d="M9 12h6"/><path d="M12 9v6"/></svg>
                Autonomo
              </button>
            </div>
            <div class="grow"></div>
            <button type="button" class="action btn-record" id="record" title="Grabar dataset">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>
              <span id="record-label">Grabar</span>
            </button>
            <button type="button" class="action btn-review" id="review" title="Abrir replayer (revision dataset)">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12l5-5v3h7a4 4 0 0 1 4 4v4"/><path d="M8 17l5 5v-3h3"/></svg>
              Revisar
            </button>
            <button type="button" class="stop" id="stop" title="Parada de emergencia">
              <svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>
              Stop
            </button>
          </div>
        </div>
      </section>

      <aside class="side">
        <div class="ctx-strip" id="ctx-strip">
          <div class="ctx" id="ctx-lat-cell"><span class="label">Latencia IA</span><span class="value" id="ctx-lat">-- ms</span></div>
          <div class="ctx" id="ctx-fps-cell"><span class="label">FPS Video</span><span class="value" id="ctx-fps">--</span></div>
          <div class="ctx" id="ctx-det-cell"><span class="label">Detecciones</span><span class="value" id="ctx-det">--</span></div>
          <div class="ctx action-default" id="ctx-action-cell"><span class="label">Accion</span><span class="value tag"><span class="dot"></span><span id="ctx-action">--</span></span></div>
        </div>

        <nav class="tabs" role="tablist">
          <button class="tab active" data-tab="telemetria" role="tab">Tele <span class="badge" id="tab-badge-tele">0</span></button>
          <button class="tab" data-tab="tuning" role="tab">Tuning <span class="badge" id="tab-badge-tuning">--</span></button>
          <button class="tab" data-tab="dataset" role="tab">Dataset <span class="badge" id="tab-badge-dataset">OFF</span></button>
          <button class="tab" data-tab="sistema" role="tab">Sistema <span class="badge" id="tab-badge-sistema">--</span></button>
        </nav>

        <div class="tab-host">
          <!-- ============== TELEMETRIA ========================================== -->
          <div class="tab-panel active" id="panel-telemetria" role="tabpanel">
            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12c4-9 14-9 18 0"/><path d="M3 12c4 9 14 9 18 0"/><circle cx="12" cy="12" r="3"/></svg></span>
                Inferencia
                <span class="tag" id="ai-tag">--</span>
              </h2>
              <div class="row"><span class="k">Backend</span><span class="v muted" id="ai-backend">--</span></div>
              <div class="row"><span class="k">Modelo</span><span class="v muted" id="ai-model">--</span></div>
              <div class="row"><span class="k">Estado</span><span class="v" id="ai-status">--</span></div>
              <div class="spark-wrap">
                <svg class="spark" id="spark-lat" viewBox="0 0 200 36" preserveAspectRatio="none">
                  <line class="grid" x1="0" y1="18" x2="200" y2="18"/>
                  <path class="area" d=""/>
                  <path d=""/>
                </svg>
                <div class="spark-data">
                  <div class="v"><span id="ai-latency">--</span><small>ms</small></div>
                  <div class="l">Latencia IA</div>
                </div>
              </div>
              <div class="spark-wrap">
                <svg class="spark cyan" id="spark-fps" viewBox="0 0 200 36" preserveAspectRatio="none">
                  <line class="grid" x1="0" y1="18" x2="200" y2="18"/>
                  <path class="area" d=""/>
                  <path d=""/>
                </svg>
                <div class="spark-data">
                  <div class="v"><span id="ai-fps">--</span><small>fps</small></div>
                  <div class="l">Video</div>
                </div>
              </div>
              <h3>Detecciones</h3>
              <div class="detections" id="detections">
                <div class="det empty"><span>Esperando inferencia...</span></div>
              </div>
            </section>

            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"/><circle cx="12" cy="12" r="6"/><path d="M9 12h6"/><path d="M12 9v6"/></svg></span>
                Autonomia
              </h2>
              <div class="row"><span class="k">Modo</span><span class="v accent" id="auto-mode">--</span></div>
              <div class="row"><span class="k">Accion</span><span class="v" id="auto-action">--</span></div>
              <div class="row"><span class="k">Carril</span><span class="v" id="auto-lane">--</span></div>
              <div class="row"><span class="k">Correccion</span><span class="v muted" id="auto-lane-correction">--</span></div>
              <div class="row"><span class="k">Senal</span><span class="v" id="auto-target">--</span></div>
              <div class="row"><span class="k">Zona / Distancia</span><span class="v muted" id="auto-zone">--</span></div>
              <div class="row"><span class="k">Motivo</span><span class="v muted" id="auto-reason">--</span></div>
            </section>

            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v18"/><path d="M4 7l8-4 8 4"/><path d="M4 17l8 4 8-4"/><path d="M4 7v10l8 4 8-4V7"/></svg></span>
                LiDAR
                <span class="tag" id="lidar-tag">--</span>
              </h2>
              <div class="row"><span class="k">Estado</span><span class="v" id="lidar-status">--</span></div>
              <div class="row"><span class="k">Distancia frontal</span><span class="v accent" id="lidar-front">--</span></div>
              <div class="row"><span class="k">Puntos</span><span class="v" id="lidar-points">--</span></div>
              <div class="row"><span class="k">Correccion</span><span class="v muted" id="lidar-correction">--</span></div>
              <div class="row"><span class="k">Motivo</span><span class="v muted" id="lidar-reason">--</span></div>
            </section>
          </div>

          <!-- ============== TUNING ============================================== -->
          <div class="tab-panel" id="panel-tuning" role="tabpanel">
            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg></span>
                Marcha
                <span class="tag" id="cruise-tag">--</span>
              </h2>
              <div class="control-block">
                <div class="readout-line">
                  <span class="value" id="cruise-value">0.650</span>
                  <span></span>
                  <span class="dir" id="cruise-dir">65%</span>
                </div>
                <label class="slider-row" for="cruise-range">
                  <span>0</span>
                  <input type="range" id="cruise-range" min="0" max="1" step="0.01" value="0.65">
                  <span>1</span>
                </label>
                <div class="input-row">
                  <input type="number" id="cruise-input" min="0" max="1" step="0.001" inputmode="decimal" value="0.650" aria-label="Velocidad de crucero">
                  <button type="button" id="cruise-base">Base</button>
                </div>
              </div>
              <h3>Pulso derecha</h3>
              <div class="control-block">
                <label class="toggle-line" for="turn-comp-enabled">
                  <span class="lbl">Activado</span>
                  <input type="checkbox" id="turn-comp-enabled">
                </label>
                <div class="compact-fields">
                  <label class="compact-field" for="turn-comp-interval">
                    <span>Cada s</span>
                    <input type="number" id="turn-comp-interval" min="0" step="0.1" inputmode="decimal" value="2.5">
                  </label>
                  <label class="compact-field" for="turn-comp-magnitude">
                    <span>Giro</span>
                    <input type="number" id="turn-comp-magnitude" min="0" max="2" step="0.01" inputmode="decimal" value="0.20">
                  </label>
                  <label class="compact-field" for="turn-comp-duration">
                    <span>Duracion</span>
                    <input type="number" id="turn-comp-duration" min="0" step="0.01" inputmode="decimal" value="0.18">
                  </label>
                </div>
                <div class="row"><span class="k">Pulso</span><span class="v accent" id="turn-comp-status">--</span></div>
                <div class="row"><span class="k">Siguiente</span><span class="v muted" id="turn-comp-next">--</span></div>
              </div>
            </section>

            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h18"/><path d="M7 8l-4 4 4 4"/><path d="M17 8l4 4-4 4"/></svg></span>
                Compensacion
                <span class="tag" id="trim-tag">--</span>
              </h2>
              <div class="control-block">
                <div class="readout-line">
                  <span class="value" id="trim-value">0.000</span>
                  <span></span>
                  <span class="dir" id="trim-dir">sin compensacion</span>
                </div>
                <label class="slider-row" for="trim-range">
                  <span>Der</span>
                  <input type="range" id="trim-range" min="-0.50" max="0.50" step="0.01" value="-0.08">
                  <span>Izq</span>
                </label>
                <div class="input-row">
                  <input type="number" id="trim-input" step="0.001" inputmode="decimal" value="-0.080" aria-label="Compensacion de giro">
                  <button type="button" id="trim-base">Base</button>
                </div>
                <div class="row"><span class="k">Giro enviado</span><span class="v accent" id="trim-effective">--</span></div>
                <div class="row"><span class="k">Giro solicitado</span><span class="v muted" id="trim-requested">--</span></div>
              </div>
            </section>

            <div class="save-defaults-row">
              <button type="button" id="save-defaults">Guardar como default</button>
              <span class="path" id="settings-path">--</span>
            </div>
          </div>

          <!-- ============== DATASET ============================================= -->
          <div class="tab-panel" id="panel-dataset" role="tabpanel">
            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg></span>
                Dataset
                <span class="tag" id="rec-tag">OFF</span>
              </h2>
              <div class="row"><span class="k">Sesion</span><span class="v muted" id="rec-session">--</span></div>
              <div class="row"><span class="k">Registros</span><span class="v" id="rec-records">0</span></div>
              <div class="row"><span class="k">Imagenes</span><span class="v" id="rec-images">0</span></div>
              <div class="row"><span class="k">Criticos</span><span class="v amber" id="rec-critical">0</span></div>
              <div class="row"><span class="k">Video</span><span class="v muted" id="rec-video">--</span></div>
              <div class="row"><span class="k">Replayer</span><span class="v muted" id="rec-replayer">--</span></div>
              <div class="row"><span class="k">Error</span><span class="v muted" id="rec-error">--</span></div>
            </section>
            <div class="empty-state">
              Usa los botones <b style="color:var(--blue)">Grabar</b> y <b style="color:var(--cyan)">Revisar</b> del deck para alternar la captura del dataset y abrir el replayer en una pestana nueva.
            </div>
          </div>

          <!-- ============== SISTEMA ============================================= -->
          <div class="tab-panel" id="panel-sistema" role="tabpanel">
            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M5 12a7 7 0 0 1 14 0"/><path d="M5 12a7 7 0 0 0 14 0"/><path d="M2 12h2"/><path d="M20 12h2"/></svg></span>
                Enlace 4G
              </h2>
              <div class="row"><span class="k">UDP</span><span class="v muted" id="link-bind">--</span></div>
              <div class="row"><span class="k">Cliente</span><span class="v" id="link-client">--</span></div>
              <div class="row"><span class="k">Ultimo paquete</span><span class="v" id="link-last">--</span></div>
              <div class="row"><span class="k">RX</span><span class="v" id="link-rx">--</span></div>
              <div class="row"><span class="k">TX</span><span class="v" id="link-tx">--</span></div>
              <div class="row"><span class="k">Errores</span><span class="v" id="link-err">0</span></div>
            </section>
            <section class="card">
              <h2>
                <span class="glyph"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="14" rx="2"/><path d="M8 20h8"/><path d="M12 18v2"/></svg></span>
                Sistema
              </h2>
              <div class="row"><span class="k">Origen control</span><span class="v" id="ctrl-source">--</span></div>
              <div class="row"><span class="k">Watchdog</span><span class="v" id="ctrl-watch">--</span></div>
              <div class="row"><span class="k">Stream activo</span><span class="v" id="stream-clients">--</span></div>
              <div class="row"><span class="k">Posts control</span><span class="v" id="control-posts">--</span></div>
            </section>
          </div>
        </div>
      </aside>
    </main>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const els = {
      pillLink: $('pill-link'),    pillLinkVal: $('pill-link-val'),
      pillVideo: $('pill-video'),  pillVideoVal: $('pill-video-val'),
      pillAi: $('pill-ai'),        pillAiVal: $('pill-ai-val'),
      pillLane: $('pill-lane'),    pillLaneVal: $('pill-lane-val'),
      pillLidar: $('pill-lidar'),  pillLidarVal: $('pill-lidar-val'),
      pillCtrl: $('pill-control'), pillCtrlVal: $('pill-control-val'),
      pillRec: $('pill-recording'), pillRecVal: $('pill-recording-val'),
      sessionClock: $('session-clock'),
      wallClock: $('wall-clock'),
      recBadge: $('rec-badge'), recBadgeText: $('rec-badge-text'),

      hudFps: $('hud-fps'), hudLat: $('hud-lat'), hudDet: $('hud-det'), hudFrame: $('hud-frame'), hudLidar: $('hud-lidar'),
      videoShell: $('video-shell'), noFeedMeta: $('no-feed-meta'),
      lidarCanvas: $('lidar-canvas'), viewCamera: $('view-camera'), viewLidar: $('view-lidar'),

      steerLeft: $('steer-left'), steerRight: $('steer-right'),
      steerVal: $('steer-val'), steerDir: $('steer-dir'),
      thrFwd: $('thr-fwd'), thrRev: $('thr-rev'),
      thrVal: $('thr-val'), thrDir: $('thr-dir'),

      modeManual: $('mode-manual'), modeAuto: $('mode-auto'), stop: $('stop'), record: $('record'), recordLabel: $('record-label'), review: $('review'),

      ctxLat: $('ctx-lat'), ctxFps: $('ctx-fps'), ctxDet: $('ctx-det'),
      ctxActionCell: $('ctx-action-cell'), ctxAction: $('ctx-action'),
      tabBadgeTele: $('tab-badge-tele'), tabBadgeTuning: $('tab-badge-tuning'),
      tabBadgeDataset: $('tab-badge-dataset'), tabBadgeSistema: $('tab-badge-sistema'),

      aiTag: $('ai-tag'),
      aiBackend: $('ai-backend'), aiModel: $('ai-model'), aiStatus: $('ai-status'),
      aiLatency: $('ai-latency'), aiFps: $('ai-fps'),
      sparkLat: $('spark-lat'), sparkFps: $('spark-fps'),
      detections: $('detections'),

      autoMode: $('auto-mode'), autoAction: $('auto-action'),
      autoLane: $('auto-lane'), autoLaneCorrection: $('auto-lane-correction'),
      autoTarget: $('auto-target'), autoZone: $('auto-zone'), autoReason: $('auto-reason'),
      lidarTag: $('lidar-tag'), lidarStatus: $('lidar-status'), lidarFront: $('lidar-front'),
      lidarPoints: $('lidar-points'), lidarCorrection: $('lidar-correction'), lidarReason: $('lidar-reason'),
      settingsPath: $('settings-path'), saveDefaults: $('save-defaults'),

      cruiseTag: $('cruise-tag'), cruiseValue: $('cruise-value'), cruiseDir: $('cruise-dir'),
      cruiseRange: $('cruise-range'), cruiseInput: $('cruise-input'), cruiseBase: $('cruise-base'),
      turnCompEnabled: $('turn-comp-enabled'),
      turnCompInterval: $('turn-comp-interval'),
      turnCompMagnitude: $('turn-comp-magnitude'),
      turnCompDuration: $('turn-comp-duration'),
      turnCompStatus: $('turn-comp-status'), turnCompNext: $('turn-comp-next'),

      trimTag: $('trim-tag'), trimValue: $('trim-value'), trimDir: $('trim-dir'),
      trimRange: $('trim-range'), trimInput: $('trim-input'), trimBase: $('trim-base'),
      trimEffective: $('trim-effective'), trimRequested: $('trim-requested'),

      recTag: $('rec-tag'), recSession: $('rec-session'), recRecords: $('rec-records'),
      recImages: $('rec-images'), recCritical: $('rec-critical'), recVideo: $('rec-video'), recReplayer: $('rec-replayer'), recError: $('rec-error'),

      linkBind: $('link-bind'), linkClient: $('link-client'), linkLast: $('link-last'),
      linkRx: $('link-rx'), linkTx: $('link-tx'), linkErr: $('link-err'),

      ctrlSource: $('ctrl-source'), ctrlWatch: $('ctrl-watch'),
      streamClients: $('stream-clients'), controlPosts: $('control-posts'),
    };

    /* state */
    const NEUTRAL_STEERING = 0.25;
    const KEY_CODES = ['w','a','s','d','x',' ','arrowup','arrowdown','arrowleft','arrowright'];
    const keys = new Set();
    let driveMode = 'manual';
    let lastSent = { steering: NEUTRAL_STEERING, throttle: 0.0 };
    let manualNeutralPosted = true;
    let trimDefault = -0.08;
    let trimPostTimer = null;
    let cruiseDefault = 0.65;
    let cruisePostTimer = null;
    let turnCompPostTimer = null;
    let activeTab = 'telemetria';
    let activeView = 'camera';
    let latestLidar = null;
    let controlSettings = {
      manual_forward_throttle: 0.60,
      manual_reverse_throttle: -0.50,
      manual_brake_throttle: -0.90,
    };

    /* sparkline buffers */
    const LAT_BUF = [], FPS_BUF = [];
    const BUF_LEN = 60;
    let lastFrames = null, lastFramesAt = null, fpsEma = 0;

    /* ---------- utilities ---------- */
    function setPillState(el, state) {
      el.classList.remove('ok','warn','bad');
      el.classList.add(state);
    }
    function setBadgeState(el, state) {
      if (!el) return;
      el.classList.remove('ok','warn','bad');
      if (state) el.classList.add(state);
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

    function setSliderPct(el, value) {
      if (!el) return;
      const min = Number(el.min);
      const max = Number(el.max);
      if (!Number.isFinite(min) || !Number.isFinite(max) || max === min) return;
      const pct = ((Number(value) - min) / (max - min)) * 100;
      el.style.setProperty('--seek-pct', clampNum(pct, 0, 100).toFixed(1) + '%');
    }

    function setTab(name) {
      activeTab = name;
      document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'panel-' + name));
    }

    function setView(name) {
      activeView = name === 'lidar' ? 'lidar' : 'camera';
      if (els.videoShell) {
        els.videoShell.classList.toggle('camera-mode', activeView === 'camera');
        els.videoShell.classList.toggle('lidar-mode', activeView === 'lidar');
      }
      if (els.viewCamera) {
        els.viewCamera.classList.toggle('active', activeView === 'camera');
        els.viewCamera.setAttribute('aria-pressed', activeView === 'camera' ? 'true' : 'false');
      }
      if (els.viewLidar) {
        els.viewLidar.classList.toggle('active', activeView === 'lidar');
        els.viewLidar.setAttribute('aria-pressed', activeView === 'lidar' ? 'true' : 'false');
      }
      drawLidarScene(latestLidar || {});
    }

    function resizeLidarCanvas() {
      const canvas = els.lidarCanvas;
      if (!canvas) return {ctx: null, w: 0, h: 0, dpr: 1};
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(320, Math.round(rect.width || 1280));
      const h = Math.max(220, Math.round(rect.height || 720));
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return {ctx, w, h, dpr};
    }

    function projectLidarPoint(point, w, h, maxRange) {
      const x = Number(point.x || 0);
      const y = Math.max(0.02, Number(point.y || 0));
      const z = Number(point.z || 0);
      const depth = clampNum(y / maxRange, 0, 1);
      const perspective = 0.34 + (1 - depth) * 0.72;
      const screenX = w * 0.5 + (x / Math.max(0.8, y + 0.75)) * w * 0.36 * perspective;
      const screenY = h * 0.86 - depth * h * 0.66 - z * h * 0.12;
      return {x: screenX, y: screenY, depth};
    }

    function drawLidarScene(lidar) {
      const canvas = els.lidarCanvas;
      if (!canvas) return;
      const {ctx, w, h} = resizeLidarCanvas();
      if (!ctx) return;
      const cfg = (lidar && lidar.config) || {};
      const maxRange = Math.max(1.0, Number(cfg.max_range_m || 8.0));
      const stopDistance = Number(cfg.stop_distance_m || 0.42);
      const slowDistance = Number(cfg.slow_distance_m || 0.85);
      const points = (lidar && lidar.points) || [];
      const safety = (lidar && lidar.safety) || {};

      ctx.clearRect(0, 0, w, h);
      const bg = ctx.createLinearGradient(0, 0, 0, h);
      bg.addColorStop(0, '#07090d');
      bg.addColorStop(1, '#030406');
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);

      ctx.save();
      ctx.translate(w * 0.5, h * 0.86);
      ctx.strokeStyle = 'rgba(78,166,255,0.16)';
      ctx.lineWidth = 1;
      for (let i = 1; i <= 8; i++) {
        const y = -i * h * 0.075;
        ctx.beginPath();
        ctx.moveTo(-w * 0.42 * (1 - i * 0.045), y);
        ctx.lineTo(w * 0.42 * (1 - i * 0.045), y);
        ctx.stroke();
      }
      for (let i = -4; i <= 4; i++) {
        ctx.beginPath();
        ctx.moveTo(i * w * 0.055, 0);
        ctx.lineTo(i * w * 0.010, -h * 0.66);
        ctx.stroke();
      }
      ctx.restore();

      function rangeLine(distance, color) {
        const depth = clampNum(distance / maxRange, 0, 1);
        const y = h * 0.86 - depth * h * 0.66;
        ctx.strokeStyle = color;
        ctx.setLineDash([7, 7]);
        ctx.beginPath();
        ctx.moveTo(w * 0.22, y);
        ctx.lineTo(w * 0.78, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      rangeLine(slowDistance, 'rgba(251,191,36,0.30)');
      rangeLine(stopDistance, 'rgba(248,113,113,0.38)');

      const sorted = points.slice().sort((a, b) => Number(b.y || 0) - Number(a.y || 0));
      for (const p of sorted) {
        const projected = projectLidarPoint(p, w, h, maxRange);
        if (projected.x < -20 || projected.x > w + 20 || projected.y < -20 || projected.y > h + 20) continue;
        const distance = Number(p.distance || Math.hypot(Number(p.x || 0), Number(p.y || 0), Number(p.z || 0)));
        const near = clampNum(1 - distance / maxRange, 0, 1);
        const radius = 1.4 + near * 4.4;
        const hue = distance <= stopDistance ? '248,113,113' : distance <= slowDistance ? '251,191,36' : '125,211,252';
        ctx.fillStyle = 'rgba(' + hue + ',' + (0.42 + near * 0.50).toFixed(2) + ')';
        ctx.beginPath();
        ctx.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.fillStyle = 'rgba(236,236,239,0.92)';
      ctx.beginPath();
      ctx.moveTo(w * 0.5, h * 0.80);
      ctx.lineTo(w * 0.47, h * 0.89);
      ctx.lineTo(w * 0.53, h * 0.89);
      ctx.closePath();
      ctx.fill();

      const status = (lidar && lidar.status) || 'searching';
      const front = safety.min_front_distance_m == null ? '--' : Number(safety.min_front_distance_m).toFixed(2) + ' m';
      ctx.fillStyle = 'rgba(8,8,12,0.72)';
      ctx.fillRect(18, h - 88, 260, 58);
      ctx.strokeStyle = status === 'stop' ? 'rgba(248,113,113,0.55)' : status === 'slow' ? 'rgba(251,191,36,0.50)' : 'rgba(78,166,255,0.36)';
      ctx.strokeRect(18.5, h - 87.5, 259, 57);
      ctx.font = '600 11px "IBM Plex Mono", monospace';
      ctx.fillStyle = '#7dd3fc';
      ctx.fillText('LIDAR ' + status.toUpperCase(), 34, h - 62);
      ctx.fillStyle = '#ececef';
      ctx.fillText(points.length + ' pts · frontal ' + front, 34, h - 40);
    }

    function renderSettings(settings) {
      if (!settings || !settings.values) return;
      Object.assign(controlSettings, settings.values);
      if (els.settingsPath) {
        els.settingsPath.textContent = (settings.persisted ? 'default · ' : 'runtime · ') + (settings.path || '--');
      }
    }

    async function saveDefaults() {
      if (trimPostTimer) {
        clearTimeout(trimPostTimer);
        trimPostTimer = null;
        await postSteeringTrim(Number(els.trimInput.value));
      }
      if (cruisePostTimer) {
        clearTimeout(cruisePostTimer);
        cruisePostTimer = null;
        await postCruiseSpeed(Number(els.cruiseInput.value));
      }
      if (turnCompPostTimer) {
        clearTimeout(turnCompPostTimer);
        turnCompPostTimer = null;
        await postTurnCompensation();
      }
      try {
        const res = await fetch('/settings/defaults', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: '{}',
          cache: 'no-store',
        });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'defaults');
        renderSettings(data.settings);
      } catch (_) {
        if (els.settingsPath) els.settingsPath.textContent = 'error guardando default';
      }
    }

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

    function clearManualKeys() {
      keys.clear();
      paintKeys();
    }

    function axisFromKeys() {
      let throttle = 0.0;
      if (keys.has('w') || keys.has('arrowup')) throttle = Number(controlSettings.manual_forward_throttle || 0.60);
      if (keys.has('s') || keys.has('arrowdown')) throttle = Number(controlSettings.manual_reverse_throttle || -0.50);
      if (keys.has('x') || keys.has(' ')) throttle = Number(controlSettings.manual_brake_throttle || -0.90);
      let steering = NEUTRAL_STEERING;
      const left = keys.has('a') || keys.has('arrowleft');
      const right = keys.has('d') || keys.has('arrowright');
      if (left && !right) steering = 1.0;
      if (right && !left) steering = -1.0;
      return { steering, throttle };
    }

    function renderControl(steering, throttle) {
      const offset = steering - NEUTRAL_STEERING;
      const leftPct  = clampNum(offset / 0.75, 0, 1);
      const rightPct = clampNum(-offset / 1.25, 0, 1);
      els.steerLeft.style.width  = (leftPct  * 50).toFixed(1) + '%';
      els.steerRight.style.width = (rightPct * 50).toFixed(1) + '%';
      els.steerVal.textContent = nfmt(steering, 2);
      els.steerDir.textContent =
        offset > 0.05  ? 'izquierda · ' + Math.round(leftPct * 100) + '%'
      : offset < -0.05 ? 'derecha · '   + Math.round(rightPct * 100) + '%'
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

    function trimDirection(trim) {
      if (trim < -0.001) return 'derecha · ' + Math.abs(trim).toFixed(3);
      if (trim > 0.001) return 'izquierda · ' + trim.toFixed(3);
      return 'sin compensacion';
    }

    function renderTrim(control) {
      const trim = Number(control.steering_trim || 0);
      trimDefault = Number(control.steering_trim_default ?? trimDefault);
      els.trimTag.textContent = nfmt(trim, 3);
      els.trimValue.textContent = nfmt(trim, 3);
      els.trimDir.textContent = trimDirection(trim);
      const clamped = clampNum(trim, Number(els.trimRange.min), Number(els.trimRange.max));
      els.trimRange.value = clamped.toFixed(2);
      setSliderPct(els.trimRange, clamped);
      if (document.activeElement !== els.trimInput) {
        els.trimInput.value = nfmt(trim, 3);
      }
      els.trimEffective.textContent = nfmt(control.effective_steering, 3);
      els.trimRequested.textContent = nfmt(control.steering, 3);
    }

    async function postSteeringTrim(trim) {
      if (!Number.isFinite(trim)) return;
      try {
        const res = await fetch('/steering-trim', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({trim}), cache: 'no-store',
        });
        if (!res.ok) throw new Error('http ' + res.status);
      } catch (_) { setPillState(els.pillCtrl, 'bad'); }
    }

    function scheduleSteeringTrim(rawValue) {
      const trim = Number(rawValue);
      if (!Number.isFinite(trim)) return;
      els.trimValue.textContent = nfmt(trim, 3);
      els.trimDir.textContent = trimDirection(trim);
      setSliderPct(els.trimRange, trim);
      if (trimPostTimer) clearTimeout(trimPostTimer);
      trimPostTimer = setTimeout(() => { trimPostTimer = null; postSteeringTrim(trim); }, 90);
    }

    function renderCruise(autonomy) {
      const cfg = (autonomy && autonomy.config) || {};
      const speed = Number(cfg.cruise_throttle ?? cruiseDefault);
      cruiseDefault = Number(cfg.cruise_throttle_default ?? cruiseDefault);
      if (!Number.isFinite(speed)) return;
      els.cruiseTag.textContent = nfmt(speed, 3);
      els.cruiseValue.textContent = nfmt(speed, 3);
      els.cruiseDir.textContent = Math.round(clampNum(speed, 0, 1) * 100) + '%';
      const clamped = clampNum(speed, Number(els.cruiseRange.min), Number(els.cruiseRange.max));
      els.cruiseRange.value = clamped.toFixed(2);
      setSliderPct(els.cruiseRange, clamped);
      if (document.activeElement !== els.cruiseInput) {
        els.cruiseInput.value = nfmt(speed, 3);
      }
    }

    async function postCruiseSpeed(speed) {
      if (!Number.isFinite(speed)) return;
      try {
        const res = await fetch('/cruise-speed', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({speed}), cache: 'no-store',
        });
        if (!res.ok) throw new Error('http ' + res.status);
      } catch (_) { setPillState(els.pillCtrl, 'bad'); }
    }

    function scheduleCruiseSpeed(rawValue) {
      const speed = Number(rawValue);
      if (!Number.isFinite(speed)) return;
      els.cruiseValue.textContent = nfmt(speed, 3);
      els.cruiseDir.textContent = Math.round(clampNum(speed, 0, 1) * 100) + '%';
      setSliderPct(els.cruiseRange, speed);
      if (cruisePostTimer) clearTimeout(cruisePostTimer);
      cruisePostTimer = setTimeout(() => { cruisePostTimer = null; postCruiseSpeed(speed); }, 90);
    }

    function renderTurnCompensation(status) {
      const tc = status || {};
      const enabled = !!tc.enabled;
      if (document.activeElement !== els.turnCompEnabled) els.turnCompEnabled.checked = enabled;
      const interval = Number(tc.interval_sec);
      const magnitude = Number(tc.magnitude);
      const duration = Number(tc.duration_sec);
      if (document.activeElement !== els.turnCompInterval && Number.isFinite(interval)) els.turnCompInterval.value = interval.toFixed(2);
      if (document.activeElement !== els.turnCompMagnitude && Number.isFinite(magnitude)) els.turnCompMagnitude.value = magnitude.toFixed(2);
      if (document.activeElement !== els.turnCompDuration && Number.isFinite(duration)) els.turnCompDuration.value = duration.toFixed(2);
      els.turnCompStatus.textContent = enabled
        ? (tc.active ? 'activo · ' + nfmt(tc.applied_correction, 3) : (tc.reason || 'esperando'))
        : 'off';
      els.turnCompNext.textContent = tc.next_in_sec == null ? '--' : nfmt(tc.next_in_sec, 2) + ' s';
    }

    function currentTurnCompPayload() {
      return {
        enabled: els.turnCompEnabled.checked,
        interval_sec: Number(els.turnCompInterval.value),
        magnitude: Number(els.turnCompMagnitude.value),
        duration_sec: Number(els.turnCompDuration.value),
      };
    }

    async function postTurnCompensation() {
      const payload = currentTurnCompPayload();
      if (!Number.isFinite(payload.interval_sec) || !Number.isFinite(payload.magnitude) || !Number.isFinite(payload.duration_sec)) return;
      try {
        const res = await fetch('/turn-compensation', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload), cache: 'no-store',
        });
        if (!res.ok) throw new Error('http ' + res.status);
      } catch (_) { setPillState(els.pillCtrl, 'bad'); }
    }

    function scheduleTurnCompensation() {
      if (turnCompPostTimer) clearTimeout(turnCompPostTimer);
      turnCompPostTimer = setTimeout(() => { turnCompPostTimer = null; postTurnCompensation(); }, 120);
    }

    /* highlight WASD on keypress */
    function paintKeys() {
      document.querySelectorAll('.key[data-key]').forEach(k => {
        const isBrake = (k.dataset.key === 's') && (keys.has(' ') || keys.has('x'));
        const pressed = keys.has(k.dataset.key)
          || (k.dataset.key === 'w' && keys.has('arrowup'))
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
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(control), cache: 'no-store',
        });
        if (!res.ok) throw new Error('http ' + res.status);
      } catch (_) { setPillState(els.pillCtrl, 'bad'); }
    }

    async function postMode(mode) {
      renderMode(mode);
      clearManualKeys();
      manualNeutralPosted = true;
      try {
        const res = await fetch('/mode', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({mode}), cache: 'no-store',
        });
        if (!res.ok) throw new Error('http ' + res.status);
      } catch (_) { setPillState(els.pillCtrl, 'bad'); }
    }

    async function toggleRecording() {
      try {
        const res = await fetch('/recording', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({action: 'toggle'}), cache: 'no-store',
        });
        if (!res.ok) throw new Error('http ' + res.status);
      } catch (_) { setPillState(els.pillRec, 'bad'); }
    }

    async function launchReplayer() {
      try {
        els.review.classList.add('active');
        const res = await fetch('/replayer/start', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: '{}', cache: 'no-store',
        });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error((data.replayer && data.replayer.last_error) || data.error || 'replayer');
        if (data.replayer && data.replayer.url) {
          window.open(data.replayer.url, 'tp2-session-replayer');
        }
      } catch (err) {
        els.recReplayer.textContent = 'error';
        els.recError.textContent = err.message || 'replayer';
      }
    }

    async function releaseManual() {
      clearManualKeys();
      lastSent = { steering: NEUTRAL_STEERING, throttle: 0.0 };
      manualNeutralPosted = true;
      renderControl(NEUTRAL_STEERING, 0.0);
      try { await fetch('/control/neutral', { method: 'POST', cache: 'no-store' }); } catch (_) {}
    }

    async function focusSafetyRelease() {
      if (driveMode === 'manual') await releaseManual();
      else clearManualKeys();
    }

    async function emergencyStop() {
      clearManualKeys();
      renderMode('manual');
      manualNeutralPosted = true;
      lastSent = { steering: NEUTRAL_STEERING, throttle: 0.0 };
      renderControl(NEUTRAL_STEERING, 0.0);
      setPillState(els.pillCtrl, 'bad');
      els.pillCtrlVal.textContent = 'OFF';
      try { await fetch('/control/stop', { method: 'POST', cache: 'no-store' }); } catch (_) {}
    }

    window.addEventListener('keydown', (event) => {
      const tag = (event.target && event.target.tagName || '').toLowerCase();
      if (['input','textarea','select'].includes(tag)) return;
      const key = event.key.toLowerCase();
      if (KEY_CODES.includes(key)) {
        event.preventDefault();
        keys.add(key);
        paintKeys();
        return;
      }
      if (key === 'm') postMode('manual');
      else if (key === 'n') postMode('autonomous');
      else if (key === 'escape') emergencyStop();
      else if (key === 'tab') {
        event.preventDefault();
        const order = ['telemetria','tuning','dataset','sistema'];
        const dir = event.shiftKey ? -1 : 1;
        const idx = order.indexOf(activeTab);
        setTab(order[(idx + dir + order.length) % order.length]);
      }
    });
    window.addEventListener('keyup', (event) => {
      keys.delete(event.key.toLowerCase());
      paintKeys();
    });
    window.addEventListener('blur', focusSafetyRelease);
    document.addEventListener('visibilitychange', () => { if (document.hidden) focusSafetyRelease(); });

    document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => setTab(t.dataset.tab)));
    els.modeManual.addEventListener('click', () => postMode('manual'));
    els.modeAuto.addEventListener('click', () => postMode('autonomous'));
    els.stop.addEventListener('click', emergencyStop);
    els.record.addEventListener('click', toggleRecording);
    els.review.addEventListener('click', launchReplayer);
    els.viewCamera.addEventListener('click', () => setView('camera'));
    els.viewLidar.addEventListener('click', () => setView('lidar'));
    window.addEventListener('resize', () => drawLidarScene(latestLidar || {}));

    els.trimRange.addEventListener('input', () => {
      els.trimInput.value = Number(els.trimRange.value).toFixed(3);
      scheduleSteeringTrim(els.trimRange.value);
    });
    els.trimInput.addEventListener('input', () => scheduleSteeringTrim(els.trimInput.value));
    els.trimInput.addEventListener('change', () => postSteeringTrim(Number(els.trimInput.value)));
    els.trimBase.addEventListener('click', () => {
      els.trimInput.value = trimDefault.toFixed(3);
      scheduleSteeringTrim(trimDefault);
    });
    els.cruiseRange.addEventListener('input', () => {
      els.cruiseInput.value = Number(els.cruiseRange.value).toFixed(3);
      scheduleCruiseSpeed(els.cruiseRange.value);
    });
    els.cruiseInput.addEventListener('input', () => scheduleCruiseSpeed(els.cruiseInput.value));
    els.cruiseInput.addEventListener('change', () => postCruiseSpeed(Number(els.cruiseInput.value)));
    els.cruiseBase.addEventListener('click', () => {
      els.cruiseInput.value = cruiseDefault.toFixed(3);
      scheduleCruiseSpeed(cruiseDefault);
    });
    els.turnCompEnabled.addEventListener('change', scheduleTurnCompensation);
    els.turnCompInterval.addEventListener('input', scheduleTurnCompensation);
    els.turnCompMagnitude.addEventListener('input', scheduleTurnCompensation);
    els.turnCompDuration.addEventListener('input', scheduleTurnCompensation);
    if (els.saveDefaults) els.saveDefaults.addEventListener('click', saveDefaults);

    /* manual control loop */
    setInterval(() => {
      if (driveMode === 'manual') {
        const c = axisFromKeys();
        const active = Math.abs(c.steering - NEUTRAL_STEERING) > 0.01 || Math.abs(c.throttle) > 0.01;
        if (active) {
          manualNeutralPosted = false;
          lastSent = c;
          renderControl(c.steering, c.throttle);
          postControl(c);
        } else {
          lastSent = { steering: NEUTRAL_STEERING, throttle: 0.0 };
          renderControl(NEUTRAL_STEERING, 0.0);
          if (!manualNeutralPosted) releaseManual();
        }
      }
    }, 50);

    /* ---------- sparklines ---------- */
    function pushBuf(buf, val) {
      buf.push(val);
      if (buf.length > BUF_LEN) buf.shift();
    }
    function drawSpark(svg, values, opts) {
      const opts2 = opts || {};
      const W = 200, H = 36;
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
        const lidar = data.lidar || {};
        const lidarSafety = lidar.safety || {};
        latestLidar = lidar;
        if (data.settings) renderSettings(data.settings);

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

        const lidarOk = !!lidar.enabled && lidar.status !== 'searching' && lidar.status !== 'stale' && lidar.status !== 'error' && (lidar.received_age_sec == null || lidar.received_age_sec < 1.5);
        if (els.videoShell) {
          const activeFeedOk = activeView === 'lidar' ? lidarOk : videoOk;
          els.videoShell.classList.toggle('no-feed', !activeFeedOk);
          if (els.noFeedMeta) {
            els.noFeedMeta.textContent = activeView === 'lidar'
              ? (lidar.enabled
                  ? 'Esperando paquete LiDAR · UDP ' + (data.udp.bind || '')
                  : 'LiDAR deshabilitado')
              : (vid.has_video
                  ? 'Cuadro retrasado · ' + (videoAge != null ? videoAge.toFixed(1) + ' s' : 'sin datos')
                  : 'Esperando cuadro de camara · UDP ' + (data.udp.bind || ''));
          }
        }

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
        if (els.ctxFps) els.ctxFps.textContent = fpsShown ? fpsShown.toFixed(1) : '--';
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
          if (els.ctxLat) els.ctxLat.textContent = inf.latency_ms + ' ms';
        } else {
          els.aiLatency.textContent = '--';
          els.hudLat.textContent = '-- ms';
          if (els.ctxLat) els.ctxLat.textContent = '-- ms';
        }
        drawSpark(els.sparkLat, LAT_BUF, { minMax: 200 });

        els.hudDet.textContent = inf.detections;
        els.hudFrame.textContent = vid.frames;
        if (els.ctxDet) els.ctxDet.textContent = inf.detections != null ? String(inf.detections) : '--';
        if (els.tabBadgeTele) els.tabBadgeTele.textContent = inf.detections != null ? String(inf.detections) : '0';

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

        /* lane assist */
        const lane = data.lane || {};
        const laneGuidance = lane.guidance || {};
        if (!lane.enabled) {
          setPillState(els.pillLane, 'warn');
          els.pillLaneVal.textContent = 'OFF';
        } else if (lane.assist_active) {
          setPillState(els.pillLane, 'ok');
          els.pillLaneVal.textContent = 'ASSIST';
        } else if (lane.usable) {
          setPillState(els.pillLane, 'ok');
          els.pillLaneVal.textContent = 'OK';
        } else if (laneGuidance.detected) {
          setPillState(els.pillLane, 'warn');
          els.pillLaneVal.textContent = 'DEBIL';
        } else {
          setPillState(els.pillLane, 'bad');
          els.pillLaneVal.textContent = 'SIN';
        }

        /* lidar */
        if (!lidar.enabled) {
          setPillState(els.pillLidar, 'warn');
          els.pillLidarVal.textContent = 'OFF';
        } else if (lidar.status === 'stop') {
          setPillState(els.pillLidar, 'bad');
          els.pillLidarVal.textContent = 'STOP';
        } else if (lidar.status === 'slow' || lidar.status === 'caution') {
          setPillState(els.pillLidar, 'warn');
          els.pillLidarVal.textContent = lidar.status === 'slow' ? 'LENTO' : 'CERCA';
        } else if (lidarOk) {
          setPillState(els.pillLidar, 'ok');
          els.pillLidarVal.textContent = 'OK';
        } else {
          setPillState(els.pillLidar, 'bad');
          els.pillLidarVal.textContent = lidar.status === 'stale' ? 'VIEJO' : 'SIN';
        }
        const frontDistance = lidarSafety.min_front_distance_m == null ? null : Number(lidarSafety.min_front_distance_m);
        els.hudLidar.textContent = frontDistance == null ? '--' : frontDistance.toFixed(2) + ' m';
        els.lidarTag.textContent = lidar.status || '--';
        els.lidarStatus.textContent = lidar.error || (lidar.status || '--');
        els.lidarFront.textContent = frontDistance == null ? '--' : frontDistance.toFixed(2) + ' m';
        els.lidarPoints.textContent = (lidar.point_count || 0) + ' · ' + (lidar.received_age_sec == null ? '--' : Number(lidar.received_age_sec).toFixed(2) + ' s');
        els.lidarCorrection.textContent = nfmt(lidarSafety.steering_correction, 3) + ' · limite ' + (lidarSafety.throttle_limit == null ? '--' : nfmt(lidarSafety.throttle_limit, 2));
        els.lidarReason.textContent = (lidar.assist_active ? 'activo · ' : '') + (lidar.assist_reason || lidarSafety.reason || '--');
        drawLidarScene(lidar);

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
        renderTrim(data.control);
        renderCruise(data.autonomy || {});
        renderTurnCompensation((data.autonomy && data.autonomy.turn_compensation) || {});

        /* autonomy card */
        els.autoMode.textContent = data.control.mode === 'autonomous' ? 'autonomo' : 'manual';
        const actionEs = {
          'continue': 'avanzar',
          'turn-left': 'girar izquierda', 'turn-right': 'girar derecha',
          'prepare-left': 'preparar izquierda', 'prepare-right': 'preparar derecha',
          'approach-stop': 'aproximar stop',
          'stop-hold': 'mantener stop',
          'confirming': 'confirmando',
          'ambiguous': 'ambiguo',
          'cooldown': 'enfriamiento',
          'speed-30': 'velocidad 30',
          'speed-90': 'velocidad 90',
          'safe-neutral': 'neutro seguro', 'crawl': 'avance lento',
          'slow': 'lento', 'cruise': 'crucero', 'stop': 'detenido',
          'brake': 'frenar',
          'lidar-stop': 'LiDAR stop', 'lidar-slow': 'LiDAR lento', 'lidar-caution': 'LiDAR cerca',
        }[autoDecision.action] || (autoDecision.action || '--');
        els.autoAction.textContent = actionEs;
        if (els.ctxAction) els.ctxAction.textContent = actionEs;
        if (els.ctxActionCell) {
          els.ctxActionCell.className = 'ctx';
          const a = autoDecision.action || '';
          if (/stop|brake|lidar-stop/.test(a)) els.ctxActionCell.classList.add('action-stop');
          else if (/turn|left|right|prepare/.test(a)) els.ctxActionCell.classList.add('action-turn');
          else if (/slow|crawl|cool|ambig|confirm|safe/.test(a)) els.ctxActionCell.classList.add('action-yield');
          else if (/continue|cruise|speed/.test(a)) els.ctxActionCell.classList.add('action-continue');
          else els.ctxActionCell.classList.add('action-default');
        }
        const tgt = autoDecision.target;
        els.autoLane.textContent = !lane.enabled
          ? 'off'
          : laneGuidance.detected
            ? ((lane.assist_active ? 'activo · ' : '') + laneGuidance.source + ' · ' + (Number(laneGuidance.confidence || 0) * 100).toFixed(0) + '%')
            : (lane.status || '--');
        const laneCorrection = lane.applied_correction != null ? lane.applied_correction : laneGuidance.correction;
        els.autoLaneCorrection.textContent =
          laneCorrection == null ? '--' : (Number(laneCorrection).toFixed(3) + ' · ' + (lane.assist_reason || laneGuidance.reason || '--'));
        els.autoTarget.textContent = tgt ? ('#' + (tgt.track_id ?? '-') + ' · ' + tgt.class + ' · ' + (Number(tgt.confidence)*100).toFixed(0) + '%') : '--';
        els.autoZone.textContent = tgt ? (tgt.zone + ' · ' + tgt.distance + ' · ' + (tgt.estimated_distance == null ? '--' : Number(tgt.estimated_distance).toFixed(2))) : '--';
        els.autoReason.textContent = (autoDecision.state ? autoDecision.state + ' · ' : '') + (autoDecision.reason || '--');

        /* recorder */
        const rec = data.recording || {};
        const recOn = !!rec.enabled;
        setPillState(els.pillRec, recOn ? 'ok' : 'warn');
        els.pillRecVal.textContent = recOn ? 'ON' : 'OFF';
        els.record.classList.toggle('active', recOn);
        if (els.recordLabel) els.recordLabel.textContent = recOn ? 'Detener' : 'Grabar';
        els.recBadge.classList.toggle('active', recOn);
        els.recBadgeText.textContent = recOn ? 'REC DATASET' : 'EN VIVO';
        els.recTag.textContent = recOn ? 'REC' : 'OFF';
        els.recSession.textContent = rec.session_dir || '--';
        els.recRecords.textContent = rec.records ?? 0;
        els.recImages.textContent = rec.images ?? 0;
        els.recCritical.textContent = rec.critical_records ?? 0;
        els.recVideo.textContent = rec.video && rec.video.enabled ? ((rec.video.frames || 0) + ' fr') : 'off';
        const replayer = data.replayer || {};
        els.review.classList.toggle('active', !!replayer.active);
        els.recReplayer.textContent = replayer.active ? ('abierto · ' + replayer.port) : (replayer.enabled ? 'listo' : 'off');
        els.recError.textContent = rec.last_error || '--';
        if (els.tabBadgeDataset) {
          els.tabBadgeDataset.textContent = recOn ? 'REC' : 'OFF';
          setBadgeState(els.tabBadgeDataset, recOn ? 'ok' : null);
        }
        if (els.tabBadgeTuning) {
          els.tabBadgeTuning.textContent = nfmt(Number(els.cruiseRange.value || 0), 2);
        }

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

        els.ctrlSource.textContent =
          data.control.source + ' · ' + nfmt(remoteSteer, 2) + ' / ' + nfmt(remoteThr, 2);
        els.ctrlWatch.textContent = nfmt(data.control.updated_age_sec, 2) + ' s';
        els.streamClients.textContent = (data.web && data.web.stream_clients) || 0;
        els.controlPosts.textContent = (data.web && data.web.control_posts) || 0;
        if (els.tabBadgeSistema) {
          const errs = (data.udp.bad_packets || 0) + (vid.decode_errors || 0);
          els.tabBadgeSistema.textContent = errs > 0 ? String(errs) : 'ok';
          setBadgeState(els.tabBadgeSistema, errs > 0 ? 'warn' : 'ok');
        }

        els.sessionClock.textContent = fmtTime(data.uptime_sec || 0);
      } catch (err) {
        setPillState(els.pillLink, 'bad'); els.pillLinkVal.textContent = 'ERR';
        setPillState(els.pillVideo, 'bad'); els.pillVideoVal.textContent = '--';
      }
    }

    pollStatus();
    setInterval(pollStatus, 250);
    setView('camera');
    renderControl(NEUTRAL_STEERING, 0.0);
    setSliderPct(els.cruiseRange, Number(els.cruiseRange.value));
    setSliderPct(els.trimRange, Number(els.trimRange.value));
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
    state.recorder.close()
    state.replayer.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
