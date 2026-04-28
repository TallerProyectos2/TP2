from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence


SIGN_CONTINUE = "OBLIGATORIO-CONTINUAR-RECTO"
SIGN_TURN_RIGHT = "OBLIGATORIO-GIRAR-DERECHA"
SIGN_TURN_LEFT = "OBLIGATORIO-GIRAR-IZQUIERDA"
SIGN_NO_ENTRY = "PROHIBIDO"
SIGN_STOP = "STOP"
SIGN_SPEED_30 = "VELOCIDAD-MAX-30"
SIGN_SPEED_90 = "VELOCIDAD-MAX-90"

KNOWN_SIGNS = {
    SIGN_CONTINUE,
    SIGN_TURN_RIGHT,
    SIGN_TURN_LEFT,
    SIGN_NO_ENTRY,
    SIGN_STOP,
    SIGN_SPEED_30,
    SIGN_SPEED_90,
}

SAFETY_SIGNS = {SIGN_STOP, SIGN_NO_ENTRY}
TURN_SIGNS = {SIGN_TURN_LEFT, SIGN_TURN_RIGHT}
SPEED_SIGNS = {SIGN_SPEED_30, SIGN_SPEED_90}

STATE_SAFE = "safe"
STATE_CRUISE = "cruise"
STATE_APPROACH = "approach"
STATE_CONFIRMING = "confirming"
STATE_STOP_HOLD = "stop-hold"
STATE_TURN_LEFT = "turn-left"
STATE_TURN_RIGHT = "turn-right"
STATE_COOLDOWN = "cooldown"
STATE_AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class AutonomousConfig:
    min_confidence: float = 0.35
    stale_prediction_sec: float = 1.25
    max_frame_age_sec: float = 1.0
    min_area_ratio: float = 0.004
    near_area_ratio: float = 0.045
    center_left: float = 0.40
    center_right: float = 0.60
    neutral_steering: float = 0.25
    neutral_throttle: float = 0.0
    crawl_throttle: float = 0.12
    slow_throttle: float = 0.18
    turn_throttle: float = 0.22
    cruise_throttle: float = 0.34
    fast_throttle: float = 0.48
    left_steering: float = 0.84
    right_steering: float = -0.84
    confirm_frames: int = 2
    safety_confirm_frames: int = 1
    max_track_age_sec: float = 1.2
    track_memory_sec: float = 0.45
    match_iou: float = 0.14
    match_center_distance: float = 0.18
    ambiguous_score_ratio: float = 0.82
    stop_hold_sec: float = 1.15
    turn_hold_sec: float = 0.75
    cooldown_sec: float = 0.85
    distance_scale: float = 0.32
    steering_rate_per_sec: float = 2.4
    throttle_rate_per_sec: float = 1.0
    brake_rate_per_sec: float = 3.0
    dry_run: bool = False


@dataclass(frozen=True)
class SignObservation:
    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    center_x: float
    center_y: float
    area_ratio: float
    zone: str
    distance: str
    estimated_distance: float | None
    lateral_offset: float
    score: float
    track_id: int | None = None
    hits: int = 1
    missed: int = 0
    age_sec: float = 0.0
    persistent: bool = False

    def to_status(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class": self.label,
            "confidence": round(self.confidence, 4),
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "center_x": round(self.center_x, 4),
            "center_y": round(self.center_y, 4),
            "lateral_offset": round(self.lateral_offset, 4),
            "area_ratio": round(self.area_ratio, 5),
            "estimated_distance": rounded(self.estimated_distance, 3),
            "zone": self.zone,
            "distance": self.distance,
            "score": round(self.score, 4),
            "hits": self.hits,
            "missed": self.missed,
            "age_sec": round(self.age_sec, 3),
            "persistent": self.persistent,
        }


@dataclass(frozen=True)
class AutonomousDecision:
    active: bool
    steering: float
    throttle: float
    action: str
    state: str
    reason: str
    target: SignObservation | None
    candidates: tuple[SignObservation, ...]
    raw_steering: float | None = None
    raw_throttle: float | None = None
    dry_run: bool = False

    def control(self) -> tuple[float, float]:
        return self.steering, self.throttle

    def to_status(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "steering": round(self.steering, 3),
            "throttle": round(self.throttle, 3),
            "raw_steering": rounded(self.raw_steering, 3),
            "raw_throttle": rounded(self.raw_throttle, 3),
            "action": self.action,
            "state": self.state,
            "reason": self.reason,
            "dry_run": self.dry_run,
            "target": None if self.target is None else self.target.to_status(),
            "candidates": [candidate.to_status() for candidate in self.candidates[:8]],
        }


@dataclass
class TrackedSign:
    track_id: int
    label: str
    observation: SignObservation
    first_seen: float
    last_seen: float
    hits: int = 1
    missed: int = 0

    def update(self, observation: SignObservation, now: float, config: AutonomousConfig) -> None:
        self.label = observation.label
        self.hits += 1
        self.missed = 0
        self.last_seen = now
        self.observation = with_track_fields(self, observation, now, config)

    def mark_missed(self, now: float, config: AutonomousConfig) -> None:
        self.missed += 1
        self.observation = with_track_fields(self, self.observation, now, config)


class SignTracker:
    def __init__(self, config: AutonomousConfig) -> None:
        self.config = config
        self.next_id = 1
        self.tracks: dict[int, TrackedSign] = {}

    def update(
        self,
        predictions: list[dict[str, Any]],
        *,
        frame_shape: Sequence[int] | None,
        now: float,
    ) -> list[SignObservation]:
        raw = build_observations(predictions, frame_shape=frame_shape, config=self.config)
        matched_tracks: set[int] = set()

        for observation in raw:
            track = self._best_match(observation, matched_tracks)
            if track is None:
                track_id = self.next_id
                self.next_id += 1
                tracked = TrackedSign(
                    track_id=track_id,
                    label=observation.label,
                    observation=observation,
                    first_seen=now,
                    last_seen=now,
                )
                tracked.observation = with_track_fields(tracked, observation, now, self.config)
                self.tracks[track_id] = tracked
                matched_tracks.add(track_id)
            else:
                track.update(observation, now, self.config)
                matched_tracks.add(track.track_id)

        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_tracks:
                track.mark_missed(now, self.config)
            if now - track.last_seen > self.config.max_track_age_sec:
                self.tracks.pop(track_id, None)

        return self.active_observations(now)

    def active_observations(self, now: float) -> list[SignObservation]:
        active: list[SignObservation] = []
        for track in self.tracks.values():
            if now - track.last_seen <= self.config.track_memory_sec:
                active.append(with_track_fields(track, track.observation, now, self.config))
        active.sort(key=lambda item: item.score, reverse=True)
        return active

    def _best_match(
        self,
        observation: SignObservation,
        used_tracks: set[int],
    ) -> TrackedSign | None:
        best: tuple[float, TrackedSign] | None = None
        for track in self.tracks.values():
            if track.track_id in used_tracks or track.label != observation.label:
                continue
            overlap = iou(track.observation, observation)
            center_distance = normalized_center_distance(track.observation, observation)
            if overlap < self.config.match_iou and center_distance > self.config.match_center_distance:
                continue
            score = overlap + max(0.0, 1.0 - center_distance)
            if best is None or score > best[0]:
                best = (score, track)
        return None if best is None else best[1]


class CommandFilter:
    def __init__(self, config: AutonomousConfig) -> None:
        self.config = config
        self.last_steering = config.neutral_steering
        self.last_throttle = config.neutral_throttle
        self.last_time: float | None = None

    def reset(self, now: float | None = None) -> None:
        self.last_steering = self.config.neutral_steering
        self.last_throttle = self.config.neutral_throttle
        self.last_time = now

    def apply(
        self,
        steering: float,
        throttle: float,
        *,
        now: float,
        urgent: bool = False,
    ) -> tuple[float, float]:
        steering = clamp(steering, -1.0, 1.0)
        throttle = clamp(throttle, -1.0, 1.0)
        if self.last_time is None or urgent:
            self.last_steering = steering
            self.last_throttle = throttle
            self.last_time = now
            return round(steering, 3), round(throttle, 3)

        dt = max(0.001, now - self.last_time)
        self.last_time = now
        steer_step = self.config.steering_rate_per_sec * dt
        throttle_rate = (
            self.config.brake_rate_per_sec
            if throttle < self.last_throttle
            else self.config.throttle_rate_per_sec
        )
        throttle_step = throttle_rate * dt
        self.last_steering = move_towards(self.last_steering, steering, steer_step)
        self.last_throttle = move_towards(self.last_throttle, throttle, throttle_step)
        return round(self.last_steering, 3), round(self.last_throttle, 3)


class AutonomousController:
    def __init__(self, config: AutonomousConfig) -> None:
        self.config = config
        self.tracker = SignTracker(config)
        self.filter = CommandFilter(config)
        self.state = STATE_SAFE
        self.state_since = 0.0
        self.last_prediction_seq: int | None = None
        self.last_observations: list[SignObservation] = []
        self.active_track_id: int | None = None
        self.stop_until = 0.0
        self.maneuver_until = 0.0
        self.cooldown_until = 0.0
        self.track_cooldowns: dict[int, float] = {}
        self.speed_cap = config.cruise_throttle

    def decide(
        self,
        predictions: list[dict[str, Any]],
        *,
        frame_shape: Sequence[int] | None,
        now: float,
        frame_time: float | None,
        predictions_time: float | None,
        prediction_seq: int | None = None,
    ) -> AutonomousDecision:
        if frame_time is None or now - frame_time > self.config.max_frame_age_sec:
            return self._safe(now, "no-fresh-frame")
        if predictions_time is None:
            return self._safe(now, "no-inference-yet")
        if now - predictions_time > self.config.stale_prediction_sec:
            return self._safe(now, "stale-inference")

        if prediction_seq is None or prediction_seq != self.last_prediction_seq:
            self.last_observations = self.tracker.update(
                predictions,
                frame_shape=frame_shape,
                now=now,
            )
            self.last_prediction_seq = prediction_seq
        else:
            self.last_observations = self.tracker.active_observations(now)

        observations = self._eligible_observations(now)
        decision = self._decide_from_observations(observations, now)
        if self.config.dry_run:
            return replace(
                decision,
                active=False,
                steering=self.config.neutral_steering,
                throttle=self.config.neutral_throttle,
                dry_run=True,
                reason=f"dry-run:{decision.reason}",
            )
        return decision

    def _eligible_observations(self, now: float) -> list[SignObservation]:
        expired = [track_id for track_id, until in self.track_cooldowns.items() if until <= now]
        for track_id in expired:
            self.track_cooldowns.pop(track_id, None)

        observations: list[SignObservation] = []
        for observation in self.last_observations:
            if observation.track_id in self.track_cooldowns and observation.label not in SAFETY_SIGNS:
                continue
            observations.append(observation)
        observations.sort(key=lambda item: item.score, reverse=True)
        return observations

    def _decide_from_observations(
        self,
        observations: list[SignObservation],
        now: float,
    ) -> AutonomousDecision:
        if self.state == STATE_STOP_HOLD and now < self.stop_until:
            target = self._target_by_id(observations, self.active_track_id)
            return self._decision(
                now,
                steering=self.config.neutral_steering,
                throttle=self.config.neutral_throttle,
                action="stop-hold",
                state=STATE_STOP_HOLD,
                reason="minimum-stop-hold",
                target=target,
                candidates=observations,
                urgent=True,
            )

        if self.state in {STATE_TURN_LEFT, STATE_TURN_RIGHT} and now < self.maneuver_until:
            steering = (
                self.config.left_steering
                if self.state == STATE_TURN_LEFT
                else self.config.right_steering
            )
            target = self._target_by_id(observations, self.active_track_id)
            return self._decision(
                now,
                steering=steering,
                throttle=min(self.config.turn_throttle, self.speed_cap),
                action=self.state,
                state=self.state,
                reason="maneuver-hold",
                target=target,
                candidates=observations,
            )

        if self.cooldown_until > now:
            return self._decision(
                now,
                steering=self.config.neutral_steering,
                throttle=min(self.config.slow_throttle, self.speed_cap),
                action="cooldown",
                state=STATE_COOLDOWN,
                reason="post-maneuver-cooldown",
                target=None,
                candidates=observations,
            )

        if not observations:
            return self._decision(
                now,
                steering=self.config.neutral_steering,
                throttle=self.speed_cap,
                action="continue",
                state=STATE_CRUISE,
                reason="no-relevant-sign",
                target=None,
                candidates=(),
            )

        ambiguous = self._ambiguous_turn_pair(observations)
        if ambiguous is not None:
            return self._decision(
                now,
                steering=self.config.neutral_steering,
                throttle=self.config.neutral_throttle,
                action="ambiguous",
                state=STATE_AMBIGUOUS,
                reason="conflicting-turn-signs",
                target=ambiguous[0],
                candidates=tuple(observations),
                urgent=True,
            )

        target = observations[0]
        if not self._is_confirmed(target):
            return self._decision(
                now,
                steering=self._steer_towards_zone(target, strength=0.08),
                throttle=min(self.config.crawl_throttle, self.speed_cap),
                action="confirming",
                state=STATE_CONFIRMING,
                reason=f"track-{target.track_id}-needs-persistence",
                target=target,
                candidates=tuple(observations),
            )

        if target.label in SAFETY_SIGNS:
            if target.distance == "far":
                return self._decision(
                    now,
                    steering=self._steer_towards_zone(target, strength=0.12),
                    throttle=min(self.config.crawl_throttle, self.speed_cap),
                    action="approach-stop",
                    state=STATE_APPROACH,
                    reason=f"{target.label}:far-{target.zone}",
                    target=target,
                    candidates=tuple(observations),
                )
            self.state = STATE_STOP_HOLD
            self.active_track_id = target.track_id
            self.stop_until = now + self.config.stop_hold_sec
            if target.track_id is not None:
                self.track_cooldowns[target.track_id] = now + self.config.stop_hold_sec + self.config.cooldown_sec
            return self._decision(
                now,
                steering=self.config.neutral_steering,
                throttle=self.config.neutral_throttle,
                action="stop",
                state=STATE_STOP_HOLD,
                reason=f"{target.label}:{target.distance}-{target.zone}",
                target=target,
                candidates=tuple(observations),
                urgent=True,
            )

        if target.label == SIGN_TURN_LEFT:
            return self._turn_decision(target, observations, now, left=True)
        if target.label == SIGN_TURN_RIGHT:
            return self._turn_decision(target, observations, now, left=False)

        if target.label == SIGN_SPEED_30:
            self.speed_cap = self.config.slow_throttle
            return self._decision(
                now,
                steering=self._steer_towards_zone(target, strength=0.05),
                throttle=self.speed_cap,
                action="speed-30",
                state=STATE_CRUISE,
                reason=f"{target.label}:{target.distance}-{target.zone}",
                target=target,
                candidates=tuple(observations),
            )

        if target.label == SIGN_SPEED_90:
            self.speed_cap = self.config.fast_throttle
            return self._decision(
                now,
                steering=self._steer_towards_zone(target, strength=0.03),
                throttle=self.speed_cap,
                action="speed-90",
                state=STATE_CRUISE,
                reason=f"{target.label}:{target.distance}-{target.zone}",
                target=target,
                candidates=tuple(observations),
            )

        self.speed_cap = max(self.speed_cap, self.config.cruise_throttle)
        return self._decision(
            now,
            steering=self._steer_towards_zone(target, strength=0.05),
            throttle=min(self.config.cruise_throttle, self.speed_cap),
            action="continue",
            state=STATE_CRUISE,
            reason=f"{target.label}:{target.distance}-{target.zone}",
            target=target,
            candidates=tuple(observations),
        )

    def _turn_decision(
        self,
        target: SignObservation,
        observations: list[SignObservation],
        now: float,
        *,
        left: bool,
    ) -> AutonomousDecision:
        strength = turn_strength(target)
        steering_target = self.config.left_steering if left else self.config.right_steering
        steering = blend(self.config.neutral_steering, steering_target, strength)
        if target.distance == "far":
            return self._decision(
                now,
                steering=steering,
                throttle=min(self.config.slow_throttle, self.speed_cap),
                action="prepare-left" if left else "prepare-right",
                state=STATE_APPROACH,
                reason=f"{target.label}:far-{target.zone}",
                target=target,
                candidates=tuple(observations),
            )

        self.state = STATE_TURN_LEFT if left else STATE_TURN_RIGHT
        self.active_track_id = target.track_id
        self.maneuver_until = now + self.config.turn_hold_sec
        self.cooldown_until = self.maneuver_until + self.config.cooldown_sec
        if target.track_id is not None:
            self.track_cooldowns[target.track_id] = self.cooldown_until
        return self._decision(
            now,
            steering=steering_target,
            throttle=min(self.config.turn_throttle, self.speed_cap),
            action="turn-left" if left else "turn-right",
            state=self.state,
            reason=f"{target.label}:{target.distance}-{target.zone}",
            target=target,
            candidates=tuple(observations),
        )

    def _decision(
        self,
        now: float,
        *,
        steering: float,
        throttle: float,
        action: str,
        state: str,
        reason: str,
        target: SignObservation | None,
        candidates: tuple[SignObservation, ...] | list[SignObservation],
        urgent: bool = False,
    ) -> AutonomousDecision:
        if state != self.state:
            self.state = state
            self.state_since = now
        raw_steering = clamp(steering, -1.0, 1.0)
        raw_throttle = clamp(throttle, -1.0, 1.0)
        filtered_steering, filtered_throttle = self.filter.apply(
            raw_steering,
            raw_throttle,
            now=now,
            urgent=urgent or raw_throttle <= self.config.neutral_throttle,
        )
        return AutonomousDecision(
            active=True,
            steering=filtered_steering,
            throttle=filtered_throttle,
            raw_steering=raw_steering,
            raw_throttle=raw_throttle,
            action=action,
            state=state,
            reason=reason,
            target=target,
            candidates=tuple(candidates),
        )

    def _safe(self, now: float, reason: str) -> AutonomousDecision:
        self.state = STATE_SAFE
        self.active_track_id = None
        self.stop_until = 0.0
        self.maneuver_until = 0.0
        self.cooldown_until = 0.0
        self.filter.reset(now)
        return AutonomousDecision(
            active=False,
            steering=round(self.config.neutral_steering, 3),
            throttle=round(self.config.neutral_throttle, 3),
            raw_steering=self.config.neutral_steering,
            raw_throttle=self.config.neutral_throttle,
            action="safe-neutral",
            state=STATE_SAFE,
            reason=reason,
            target=None,
            candidates=tuple(self.last_observations[:8]),
            dry_run=self.config.dry_run,
        )

    def _is_confirmed(self, observation: SignObservation) -> bool:
        required = (
            self.config.safety_confirm_frames
            if observation.label in SAFETY_SIGNS
            else self.config.confirm_frames
        )
        return observation.hits >= max(1, required)

    def _ambiguous_turn_pair(
        self,
        observations: list[SignObservation],
    ) -> tuple[SignObservation, SignObservation] | None:
        turns = [
            item
            for item in observations
            if item.label in TURN_SIGNS and item.distance != "far" and self._is_confirmed(item)
        ]
        if len(turns) < 2:
            return None
        first = turns[0]
        for second in turns[1:]:
            if first.label != second.label and second.score >= first.score * self.config.ambiguous_score_ratio:
                return first, second
        return None

    def _target_by_id(
        self,
        observations: list[SignObservation],
        track_id: int | None,
    ) -> SignObservation | None:
        if track_id is None:
            return None
        for observation in observations:
            if observation.track_id == track_id:
                return observation
        return None

    def _steer_towards_zone(self, observation: SignObservation, *, strength: float) -> float:
        if observation.zone == "left":
            return blend(self.config.neutral_steering, self.config.left_steering, strength)
        if observation.zone == "right":
            return blend(self.config.neutral_steering, self.config.right_steering, strength)
        return self.config.neutral_steering


def decide_autonomous_control(
    predictions: list[dict[str, Any]],
    *,
    frame_shape: Sequence[int] | None,
    now: float,
    frame_time: float | None,
    predictions_time: float | None,
    config: AutonomousConfig,
    prediction_seq: int | None = None,
) -> AutonomousDecision:
    controller = AutonomousController(config)
    return controller.decide(
        predictions,
        frame_shape=frame_shape,
        now=now,
        frame_time=frame_time,
        predictions_time=predictions_time,
        prediction_seq=prediction_seq,
    )


def build_observations(
    predictions: list[dict[str, Any]],
    *,
    frame_shape: Sequence[int] | None,
    config: AutonomousConfig,
) -> list[SignObservation]:
    frame_h, frame_w = frame_size(frame_shape)
    if frame_w <= 0 or frame_h <= 0:
        return []

    observations: list[SignObservation] = []
    for prediction in predictions:
        observation = observation_from_prediction(prediction, frame_w, frame_h, config)
        if observation is not None:
            observations.append(observation)

    observations.sort(key=lambda item: item.score, reverse=True)
    return observations


def observation_from_prediction(
    prediction: dict[str, Any],
    frame_w: int,
    frame_h: int,
    config: AutonomousConfig,
) -> SignObservation | None:
    label = str(prediction.get("class") or prediction.get("class_name") or "").strip()
    if label not in KNOWN_SIGNS:
        return None

    confidence = float_or_none(prediction.get("confidence"))
    if confidence is None or confidence < config.min_confidence:
        return None

    x = float_or_none(prediction.get("x"))
    y = float_or_none(prediction.get("y"))
    width = float_or_none(prediction.get("width"))
    height = float_or_none(prediction.get("height"))
    if x is None or y is None or width is None or height is None:
        return None
    if width <= 0 or height <= 0:
        return None

    center_x = clamp(x / frame_w, 0.0, 1.0)
    center_y = clamp(y / frame_h, 0.0, 1.0)
    area_ratio = clamp((width * height) / float(frame_w * frame_h), 0.0, 1.0)
    if area_ratio < config.min_area_ratio:
        return None

    zone = zone_for(center_x, config)
    distance = distance_bucket(area_ratio, config)
    estimated_distance = (
        None
        if area_ratio <= 0
        else config.distance_scale / math.sqrt(max(area_ratio, 0.000001))
    )
    lateral_offset = center_x - 0.5
    class_weight = {
        SIGN_STOP: 1.80,
        SIGN_NO_ENTRY: 1.72,
        SIGN_TURN_LEFT: 1.28,
        SIGN_TURN_RIGHT: 1.28,
        SIGN_SPEED_30: 1.14,
        SIGN_SPEED_90: 1.02,
        SIGN_CONTINUE: 0.96,
    }[label]
    centrality = 1.0 - min(1.0, abs(lateral_offset) * 2.0)
    zone_weight = 0.78 + 0.22 * centrality
    lower_frame_weight = 0.78 + 0.22 * center_y
    distance_weight = min(2.5, max(0.15, (area_ratio / max(config.near_area_ratio, 0.0001)) * 2.5))
    score = confidence * class_weight * zone_weight * lower_frame_weight * distance_weight

    return SignObservation(
        label=label,
        confidence=confidence,
        x=x,
        y=y,
        width=width,
        height=height,
        center_x=center_x,
        center_y=center_y,
        area_ratio=area_ratio,
        zone=zone,
        distance=distance,
        estimated_distance=estimated_distance,
        lateral_offset=lateral_offset,
        score=score,
    )


def with_track_fields(
    track: TrackedSign,
    observation: SignObservation,
    now: float,
    config: AutonomousConfig,
) -> SignObservation:
    required = config.safety_confirm_frames if track.label in SAFETY_SIGNS else config.confirm_frames
    persistence_weight = 1.0 + min(0.35, max(0, track.hits - 1) * 0.08)
    stale_penalty = max(0.55, 1.0 - track.missed * 0.18)
    return replace(
        observation,
        track_id=track.track_id,
        hits=track.hits,
        missed=track.missed,
        age_sec=max(0.0, now - track.first_seen),
        persistent=track.hits >= max(1, required),
        score=observation.score * persistence_weight * stale_penalty,
    )


def iou(a: SignObservation, b: SignObservation) -> float:
    ax1, ay1, ax2, ay2 = bbox_corners(a)
    bx1, by1, bx2, by2 = bbox_corners(b)
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


def bbox_corners(observation: SignObservation) -> tuple[float, float, float, float]:
    return (
        observation.x - observation.width / 2.0,
        observation.y - observation.height / 2.0,
        observation.x + observation.width / 2.0,
        observation.y + observation.height / 2.0,
    )


def normalized_center_distance(a: SignObservation, b: SignObservation) -> float:
    return math.hypot(a.center_x - b.center_x, a.center_y - b.center_y)


def frame_size(frame_shape: Sequence[int] | None) -> tuple[int, int]:
    if frame_shape is None or len(frame_shape) < 2:
        return 0, 0
    try:
        return int(frame_shape[0]), int(frame_shape[1])
    except (TypeError, ValueError):
        return 0, 0


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def move_towards(current: float, target: float, step: float) -> float:
    if abs(target - current) <= step:
        return target
    return current + step if target > current else current - step


def zone_for(center_x: float, config: AutonomousConfig) -> str:
    if center_x < config.center_left:
        return "left"
    if center_x > config.center_right:
        return "right"
    return "center"


def distance_bucket(area_ratio: float, config: AutonomousConfig) -> str:
    if area_ratio >= config.near_area_ratio:
        return "near"
    if area_ratio >= config.near_area_ratio * 0.40:
        return "mid"
    return "far"


def turn_strength(observation: SignObservation) -> float:
    if observation.distance == "near":
        return 1.0
    if observation.distance == "mid":
        return 0.72
    return 0.36


def blend(start: float, end: float, factor: float) -> float:
    return start + (end - start) * clamp(factor, 0.0, 1.0)


def rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)
