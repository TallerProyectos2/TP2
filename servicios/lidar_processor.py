from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class LidarConfig:
    enabled: bool = True
    stale_sec: float = 0.75
    min_range_m: float = 0.05
    max_range_m: float = 8.0
    front_angle_deg: float = 22.5
    front_point_count: int = 45
    side_angle_deg: float = 82.0
    stop_distance_m: float = 0.15
    slow_distance_m: float = 0.85
    caution_distance_m: float = 1.35
    slow_throttle: float = 0.25
    avoidance_gain: float = 0.55
    max_steering_correction: float = 0.45
    center_deadband_m: float = 0.08
    max_status_points: int = 720


@dataclass(frozen=True)
class LidarPoint:
    x: float
    y: float
    z: float
    intensity: float | None = None
    beam_index: int | None = None

    @property
    def distance(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan2(self.x, self.y))

    def to_status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "distance": round(self.distance, 3),
            "angle_deg": round(self.angle_deg, 2),
        }
        if self.intensity is not None:
            payload["intensity"] = round(self.intensity, 4)
        if self.beam_index is not None:
            payload["beam_index"] = self.beam_index
        return payload


@dataclass(frozen=True)
class LidarScan:
    points: tuple[LidarPoint, ...]
    received_at: float
    source: str = "car"
    frame_id: str = "tp2-lidar"
    sensor_time: float | None = None
    sample_count: int | None = None
    angle_increment_rad: float | None = None

    def age(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        return max(0.0, now - self.received_at)


@dataclass(frozen=True)
class LidarSafety:
    status: str
    reason: str
    active: bool
    clear: bool
    point_count: int
    age_sec: float | None
    nearest: LidarPoint | None
    min_front_distance_m: float | None
    steering_correction: float
    throttle_limit: float | None
    front_angle_deg: float | None = None
    front_point_count: int | None = None
    front_obstacle_count: int = 0

    def to_status(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "active": self.active,
            "clear": self.clear,
            "point_count": self.point_count,
            "age_sec": None if self.age_sec is None else round(self.age_sec, 3),
            "nearest": None if self.nearest is None else self.nearest.to_status(),
            "min_front_distance_m": (
                None if self.min_front_distance_m is None else round(self.min_front_distance_m, 3)
            ),
            "steering_correction": round(self.steering_correction, 3),
            "throttle_limit": None if self.throttle_limit is None else round(self.throttle_limit, 3),
            "front_angle_deg": (
                None if self.front_angle_deg is None else round(self.front_angle_deg, 2)
            ),
            "front_point_count": self.front_point_count,
            "front_obstacle_count": self.front_obstacle_count,
        }


def normalize_lidar_payload(
    payload: Any,
    *,
    config: LidarConfig | None = None,
    received_at: float | None = None,
) -> LidarScan:
    config = config or LidarConfig()
    received_at = time.time() if received_at is None else received_at
    source = "car"
    frame_id = "tp2-lidar"
    sensor_time: float | None = None

    payload = _decode_payload_container(payload)
    points_payload: Any = payload
    if isinstance(payload, dict):
        source = str(payload.get("source") or payload.get("sensor") or source)
        frame_id = str(payload.get("frame_id") or payload.get("frame") or frame_id)
        sensor_time = _finite_float(payload.get("timestamp", payload.get("ts", payload.get("time"))))
        if "ranges" in payload:
            points, sample_count, angle_increment_rad = _points_from_ranges(payload, config)
            return LidarScan(
                tuple(points),
                received_at=received_at,
                source=source,
                frame_id=frame_id,
                sensor_time=sensor_time,
                sample_count=sample_count,
                angle_increment_rad=angle_increment_rad,
            )
        for key in ("points", "point_cloud", "pointcloud", "scan"):
            if key in payload:
                points_payload = payload[key]
                break

    points, sample_count, angle_increment_rad = _points_from_iterable(points_payload, config)
    return LidarScan(
        tuple(points),
        received_at=received_at,
        source=source,
        frame_id=frame_id,
        sensor_time=sensor_time,
        sample_count=sample_count,
        angle_increment_rad=angle_increment_rad,
    )


def analyze_lidar_scan(
    scan: LidarScan | None,
    *,
    config: LidarConfig,
    now: float | None = None,
) -> LidarSafety:
    now = time.time() if now is None else now
    if not config.enabled:
        return LidarSafety("disabled", "disabled", False, True, 0, None, None, None, 0.0, None)
    if scan is None:
        return LidarSafety("searching", "no-scan", False, True, 0, None, None, None, 0.0, None)

    age = scan.age(now)
    point_count = len(scan.points)
    if age > config.stale_sec:
        return LidarSafety("stale", "stale-scan", False, True, point_count, age, None, None, 0.0, None)
    if point_count == 0:
        return LidarSafety("empty", "empty-scan", False, False, 0, age, None, None, 0.0, None)

    front_angle_deg = _effective_front_angle_deg(scan, config)
    front_points = [
        point
        for point in scan.points
        if point.y > 0.0 and abs(point.angle_deg) <= front_angle_deg
    ]
    if not front_points:
        return LidarSafety(
            "clear",
            "no-front-points",
            False,
            True,
            point_count,
            age,
            None,
            None,
            0.0,
            None,
            front_angle_deg,
            config.front_point_count,
            0,
        )

    nearest = min(front_points, key=lambda point: point.distance)
    distance = nearest.distance
    if distance <= config.stop_distance_m + 1e-6:
        return LidarSafety(
            "stop",
            "front-obstacle-stop",
            True,
            False,
            point_count,
            age,
            nearest,
            distance,
            0.0,
            0.0,
            front_angle_deg,
            config.front_point_count,
            len(front_points),
        )

    if distance <= config.slow_distance_m + 1e-6:
        correction = avoidance_correction(scan.points, nearest, config)
        return LidarSafety(
            "slow",
            "front-obstacle-slow",
            True,
            False,
            point_count,
            age,
            nearest,
            distance,
            correction,
            config.slow_throttle,
            front_angle_deg,
            config.front_point_count,
            len(front_points),
        )

    if distance <= config.caution_distance_m + 1e-6:
        correction = avoidance_correction(scan.points, nearest, config) * 0.5
        return LidarSafety(
            "caution",
            "front-obstacle-caution",
            True,
            False,
            point_count,
            age,
            nearest,
            distance,
            correction,
            None,
            front_angle_deg,
            config.front_point_count,
            len(front_points),
        )

    return LidarSafety(
        "clear",
        "front-clear",
        False,
        True,
        point_count,
        age,
        nearest,
        distance,
        0.0,
        None,
        front_angle_deg,
        config.front_point_count,
        len(front_points),
    )


def lidar_status_points(scan: LidarScan | None, config: LidarConfig) -> list[dict[str, Any]]:
    if scan is None or config.max_status_points <= 0:
        return []
    points = scan.points
    if len(points) <= config.max_status_points:
        selected = points
    else:
        step = max(1, math.ceil(len(points) / config.max_status_points))
        selected = points[::step][: config.max_status_points]
    return [point.to_status() for point in selected]


def avoidance_correction(
    points: Sequence[LidarPoint],
    nearest: LidarPoint,
    config: LidarConfig,
) -> float:
    clearance_left = _sector_clearance(points, -config.side_angle_deg, -config.front_angle_deg)
    clearance_right = _sector_clearance(points, config.front_angle_deg, config.side_angle_deg)
    if abs(nearest.x) <= config.center_deadband_m:
        steer_sign = 1.0 if clearance_left >= clearance_right else -1.0
    else:
        steer_sign = 1.0 if nearest.x > 0.0 else -1.0

    span = max(0.01, config.slow_distance_m - config.stop_distance_m)
    proximity = _clamp((config.slow_distance_m - nearest.distance) / span, 0.0, 1.0)
    magnitude = _clamp(
        proximity * config.avoidance_gain,
        0.0,
        config.max_steering_correction,
    )
    return round(steer_sign * magnitude, 3)


def _sector_clearance(points: Sequence[LidarPoint], angle_min: float, angle_max: float) -> float:
    distances = [
        point.distance
        for point in points
        if point.y > 0.0 and angle_min <= point.angle_deg <= angle_max
    ]
    return max(distances) if distances else float("inf")


def _decode_payload_container(payload: Any) -> Any:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        raw = bytes(payload).strip()
        if not raw:
            return []
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return payload
    return payload


def _points_from_ranges(
    payload: dict[str, Any],
    config: LidarConfig,
) -> tuple[list[LidarPoint], int, float | None]:
    ranges = payload.get("ranges", [])
    angles = payload.get("angles")
    unit = str(payload.get("angle_unit") or payload.get("unit") or "rad").strip().lower()
    angle_min = _finite_float(payload.get("angle_min", payload.get("start_angle")))
    angle_increment = _finite_float(payload.get("angle_increment", payload.get("angle_step")))
    angle_max = _finite_float(payload.get("angle_max", payload.get("end_angle")))

    range_values = list(_coerce_range_sequence(ranges))
    sample_count = len(range_values)
    if angles is not None:
        angle_values = list(_coerce_numeric_sequence(angles))
    else:
        if angle_min is None:
            angle_min = 0.0
        if angle_increment is None:
            if angle_max is not None and len(range_values) > 1:
                angle_increment = (angle_max - angle_min) / (len(range_values) - 1)
            else:
                angle_increment = math.radians(1.0)
        angle_values = [angle_min + idx * angle_increment for idx in range(len(range_values))]

    if unit in {"deg", "degree", "degrees"}:
        angle_values = [math.radians(value) for value in angle_values]
        if angle_increment is not None:
            angle_increment = math.radians(angle_increment)

    points: list[LidarPoint] = []
    intensities = list(_coerce_numeric_sequence(payload.get("intensities", [])))
    for idx, distance in enumerate(range_values):
        if idx >= len(angle_values):
            break
        if distance is None:
            continue
        point = _point_from_polar(
            angle_values[idx],
            distance,
            intensities[idx] if idx < len(intensities) else None,
            config,
            beam_index=idx,
        )
        if point is not None:
            points.append(point)
    return points, sample_count, angle_increment


def _points_from_iterable(
    payload: Any,
    config: LidarConfig,
) -> tuple[list[LidarPoint], int | None, float | None]:
    if isinstance(payload, np.ndarray):
        values = payload.tolist()
    else:
        values = payload
    if isinstance(values, dict):
        values = values.values()
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, bytearray, memoryview)):
        return [], None, None

    items = list(values)
    flat_ranges = list(_coerce_range_sequence(items))
    if items and all(_is_scalar_range_token(item) for item in items):
        angle_increment = (2.0 * math.pi) / max(1, len(flat_ranges))
        points = [
            point
            for idx, distance in enumerate(flat_ranges)
            if distance is not None
            and (point := _point_from_polar(idx * angle_increment, distance, None, config, beam_index=idx)) is not None
        ]
        return points, len(flat_ranges), angle_increment

    points: list[LidarPoint] = []
    for item in items:
        point = _point_from_item(item, config)
        if point is not None:
            points.append(point)
    return points, None, None


def _point_from_item(item: Any, config: LidarConfig) -> LidarPoint | None:
    if isinstance(item, dict):
        if "range" in item or "distance" in item:
            distance = _finite_float(item.get("range", item.get("distance")))
            angle = _finite_float(item.get("angle", item.get("angle_rad", item.get("theta", item.get("angle_deg")))))
            if distance is not None and angle is not None:
                unit = str(item.get("angle_unit") or item.get("unit") or "rad").strip().lower()
                if unit in {"deg", "degree", "degrees"} or "angle_deg" in item:
                    angle = math.radians(angle)
                return _point_from_polar(angle, distance, _finite_float(item.get("intensity")), config)
        x = _finite_float(item.get("x", item.get("right", item.get("right_m"))))
        y = _finite_float(item.get("y", item.get("forward", item.get("forward_m"))))
        z = _finite_float(item.get("z", item.get("up", item.get("height", 0.0)))) or 0.0
        intensity = _finite_float(item.get("intensity", item.get("r")))
        return _point_from_cartesian(x, y, z, intensity, config)

    if isinstance(item, np.ndarray):
        item = item.tolist()
    if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray, memoryview)):
        values = [_finite_float(value) for value in item[:4]]
        if len(values) >= 2 and values[0] is not None and values[1] is not None:
            x = values[0]
            y = values[1]
            z = values[2] if len(values) >= 3 and values[2] is not None else 0.0
            intensity = values[3] if len(values) >= 4 else None
            return _point_from_cartesian(x, y, z, intensity, config)
    return None


def _point_from_polar(
    angle_rad: float,
    distance: float,
    intensity: float | None,
    config: LidarConfig,
    *,
    beam_index: int | None = None,
) -> LidarPoint | None:
    if not math.isfinite(angle_rad):
        return None
    if not _range_ok(distance, config):
        return None
    x = math.sin(angle_rad) * distance
    y = math.cos(angle_rad) * distance
    return LidarPoint(x, y, 0.0, intensity, beam_index)


def _point_from_cartesian(
    x: float | None,
    y: float | None,
    z: float,
    intensity: float | None,
    config: LidarConfig,
) -> LidarPoint | None:
    if x is None or y is None:
        return None
    point = LidarPoint(float(x), float(y), float(z), intensity)
    return point if _range_ok(point.distance, config) else None


def _range_ok(distance: float, config: LidarConfig) -> bool:
    return math.isfinite(distance) and config.min_range_m <= distance <= config.max_range_m


def _effective_front_angle_deg(scan: LidarScan, config: LidarConfig) -> float:
    sample_count = scan.sample_count or 0
    front_point_count = max(0, int(config.front_point_count or 0))
    if sample_count > 0 and front_point_count > 0:
        half_window = (min(front_point_count, sample_count) * 360.0 / sample_count) / 2.0
        return _clamp(half_window, 0.0, 180.0)
    return _clamp(float(config.front_angle_deg), 0.0, 180.0)


def _coerce_numeric_sequence(values: Any) -> Iterable[float]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.reshape(-1).tolist()
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, bytearray, memoryview)):
        return []
    clean: list[float] = []
    for value in values:
        number = _finite_float(value)
        if number is not None:
            clean.append(number)
    return clean


def _coerce_range_sequence(values: Any) -> Iterable[float | None]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        values = values.reshape(-1).tolist()
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, bytearray, memoryview)):
        return []
    clean: list[float | None] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            clean.append(None)
            continue
        clean.append(number if math.isfinite(number) else None)
    return clean


def _is_scalar_range_token(value: Any) -> bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, bytes, bytearray, memoryview, dict)):
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
