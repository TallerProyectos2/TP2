from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from autonomous_driver import AutonomousDecision
from coche import (
    CriticalFrameAnalyzer,
    NEUTRAL_STEERING,
    NEUTRAL_THROTTLE,
    RuntimeState,
    SessionRecorder,
    STEERING_TRIM,
    corrected_steering,
)


def prediction(label="STOP", confidence=0.9, x=100, y=100, width=40, height=40):
    return {
        "class": label,
        "confidence": confidence,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def decision(action="continue", state="cruise", reason="test"):
    return AutonomousDecision(
        active=True,
        steering=NEUTRAL_STEERING,
        throttle=0.5,
        raw_steering=NEUTRAL_STEERING,
        raw_throttle=0.5,
        action=action,
        state=state,
        reason=reason,
        target=None,
        candidates=(),
    )


def analyzer():
    return CriticalFrameAnalyzer(
        low_confidence_min=0.35,
        low_confidence_max=0.55,
        disappear_frames=3,
        match_iou=0.1,
        match_center_distance=0.2,
    )


class RuntimeStateModeTest(unittest.TestCase):
    def test_web_control_does_not_exit_autonomous_mode(self):
        state = RuntimeState()
        state.set_drive_mode("autonomous")

        control = state.set_control(-0.8, 0.6, source="web")

        self.assertEqual(control["mode"], "autonomous")
        self.assertNotEqual(control["source"], "web")

    def test_manual_control_applies_in_manual_mode(self):
        state = RuntimeState()
        state.set_drive_mode("manual")

        control = state.set_control(-0.8, 0.6, source="web")

        self.assertEqual(control["mode"], "manual")
        self.assertTrue(control["armed"])
        self.assertEqual(control["source"], "web")
        self.assertEqual(control["steering"], -0.8)
        self.assertEqual(control["throttle"], 0.6)

    def test_neutral_manual_control_does_not_arm(self):
        state = RuntimeState()
        state.set_drive_mode("manual")

        control = state.set_control(NEUTRAL_STEERING, NEUTRAL_THROTTLE, source="web")

        self.assertEqual(control["mode"], "manual")
        self.assertFalse(control["armed"])
        self.assertEqual(control["source"], "neutral")

    def test_manual_release_preserves_autonomous_mode(self):
        state = RuntimeState()
        state.set_drive_mode("autonomous")

        control = state.release_manual_control("neutral")

        self.assertEqual(control["mode"], "autonomous")
        self.assertNotEqual(control["source"], "neutral")

    def test_explicit_stop_exits_autonomous_mode(self):
        state = RuntimeState()
        state.set_drive_mode("autonomous")

        control = state.neutral("stop")

        self.assertEqual(control["mode"], "manual")
        self.assertFalse(control["armed"])
        self.assertEqual(control["source"], "stop")
        self.assertEqual(control["steering"], NEUTRAL_STEERING)
        self.assertEqual(control["throttle"], NEUTRAL_THROTTLE)

    def test_control_snapshot_exposes_steering_trim(self):
        state = RuntimeState()

        control = state.control_snapshot_locked()

        self.assertEqual(control["steering_trim"], STEERING_TRIM)
        self.assertEqual(control["steering_trim_default"], STEERING_TRIM)
        self.assertEqual(
            control["effective_steering"],
            corrected_steering(NEUTRAL_STEERING, control["applied_steering_trim"]),
        )
        self.assertLess(control["effective_steering"], control["steering"])

    def test_steering_trim_can_be_changed_live(self):
        state = RuntimeState()

        control = state.set_steering_trim(0.125)

        self.assertEqual(control["steering_trim"], 0.125)
        self.assertEqual(
            control["effective_steering"],
            corrected_steering(NEUTRAL_STEERING, 0.125),
        )

    def test_steering_trim_is_bypassed_for_open_turns(self):
        state = RuntimeState()
        with state.lock:
            state.drive_mode = "autonomous"
            state.steering = 1.0
            state.autonomous_decision = decision(action="turn-left", state="turn-left")
            control = state.control_snapshot_locked()

        self.assertEqual(control["steering_trim"], STEERING_TRIM)
        self.assertEqual(control["applied_steering_trim"], 0.0)
        self.assertTrue(control["steering_trim_bypassed"])
        self.assertEqual(control["effective_steering"], 1.0)

    def test_rejects_invalid_live_steering_trim(self):
        state = RuntimeState()

        with self.assertRaises(ValueError):
            state.set_steering_trim("nan")

    def test_cruise_speed_can_be_changed_live(self):
        state = RuntimeState()

        status = state.set_cruise_speed(0.42)
        config = status["autonomy"]["config"]

        self.assertEqual(config["cruise_throttle"], 0.42)
        self.assertEqual(config["crawl_throttle"], 0.42)
        self.assertEqual(config["slow_throttle"], 0.42)
        self.assertEqual(config["turn_throttle"], 0.42)
        self.assertEqual(config["fast_throttle"], 0.42)

    def test_rejects_invalid_live_cruise_speed(self):
        state = RuntimeState()

        with self.assertRaises(ValueError):
            state.set_cruise_speed("nan")

    def test_turn_compensation_applies_right_pulse_to_forward_actions(self):
        state = RuntimeState()
        state.set_turn_compensation(enabled=True, interval_sec=0.1, magnitude=0.2, duration_sec=0.05)
        with state.lock:
            state.drive_mode = "autonomous"
            state.turn_compensation_last_pulse_at = 0.0
            adjusted = state._apply_turn_compensation_locked(decision(), 1.0)
            pulse = state.turn_compensation_snapshot_locked(now=1.0)

        self.assertEqual(adjusted.steering, 0.05)
        self.assertEqual(adjusted.raw_steering, 0.05)
        self.assertIn("turn-comp=-0.200", adjusted.reason)
        self.assertTrue(pulse["active"])
        self.assertEqual(pulse["applied_correction"], -0.2)

    def test_turn_compensation_does_not_compete_with_open_turns(self):
        state = RuntimeState()
        state.set_turn_compensation(enabled=True, interval_sec=0.1, magnitude=0.2, duration_sec=0.05)
        turn = decision(
            action="turn-right",
            state="turn-right",
            reason="turn-test",
        )
        with state.lock:
            state.drive_mode = "autonomous"
            state.turn_compensation_last_pulse_at = 0.0
            adjusted = state._apply_turn_compensation_locked(turn, 1.0)

        self.assertEqual(adjusted.steering, turn.steering)
        self.assertFalse(state.turn_compensation_active)
        self.assertEqual(state.turn_compensation_reason, "action-turn-right")

    def test_runtime_settings_update_and_save_defaults(self):
        state = RuntimeState()
        with tempfile.TemporaryDirectory() as tmp:
            state.settings_path = Path(tmp) / "defaults.json"

            settings = state.update_runtime_settings(
                {
                    "values": {
                        "steering_trim": -0.31,
                        "cruise_throttle": 0.44,
                        "turn_pulse_enabled": False,
                        "turn_compensation_enabled": True,
                        "turn_compensation_interval_sec": 0.5,
                        "turn_compensation_duration_sec": 0.2,
                        "turn_compensation_magnitude": 0.33,
                    }
                }
            )
            saved = state.save_current_settings_as_defaults()

            self.assertEqual(settings["values"]["steering_trim"], -0.31)
            self.assertEqual(settings["values"]["cruise_throttle"], 0.44)
            self.assertFalse(settings["values"]["turn_pulse_enabled"])
            self.assertTrue(settings["values"]["turn_compensation_enabled"])
            self.assertEqual(settings["values"]["turn_compensation_interval_sec"], 0.5)
            self.assertEqual(settings["values"]["turn_compensation_duration_sec"], 0.2)
            self.assertEqual(settings["values"]["turn_compensation_magnitude"], 0.33)
            self.assertTrue(saved["persisted"])
            self.assertTrue(state.settings_path.exists())

    def test_autonomous_mode_applies_lane_correction_when_cruising(self):
        state = RuntimeState()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (22, 22, 24)
        cv2.line(frame, (300, 440), (320, 180), (220, 220, 35), 14, cv2.LINE_AA)
        cv2.line(frame, (570, 440), (520, 180), (220, 220, 35), 14, cv2.LINE_AA)

        seq = state.update_frame(frame)
        state.set_predictions(seq, [], 1)
        result = state.set_drive_mode("autonomous")

        self.assertTrue(result["control"]["armed"])
        self.assertLess(result["control"]["steering"], NEUTRAL_STEERING)
        self.assertEqual(result["control"]["throttle"], 0.35)
        lane = state.snapshot()["lane"]
        self.assertTrue(lane["assist_active"])
        self.assertLess(lane["applied_correction"], 0.0)
        self.assertIn("recovery", lane["assist_reason"])

    def test_lane_assist_does_not_compete_with_open_turns(self):
        state = RuntimeState()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (22, 22, 24)
        cv2.line(frame, (300, 440), (320, 180), (220, 220, 35), 14, cv2.LINE_AA)
        cv2.line(frame, (570, 440), (520, 180), (220, 220, 35), 14, cv2.LINE_AA)
        state.update_frame(frame)
        state.set_drive_mode("autonomous")
        turn = decision(
            action="turn-right",
            state="turn-right",
            reason="turn-test",
        )

        adjusted = state._apply_lane_assist_locked(turn, 1.0)

        self.assertEqual(adjusted.steering, turn.steering)
        self.assertFalse(state.lane_assist_active)
        self.assertEqual(state.lane_assist_reason, "action-turn-right")


class CriticalFrameAnalyzerTest(unittest.TestCase):
    def test_low_confidence_and_class_change_are_flagged(self):
        detector = analyzer()
        detector.evaluate(
            frame_seq=1,
            frame_shape=(240, 320, 3),
            predictions=[prediction("STOP", confidence=0.8)],
            decision=decision(),
            operator_events=[],
        )

        enriched, flags = detector.evaluate(
            frame_seq=2,
            frame_shape=(240, 320, 3),
            predictions=[prediction("VELOCIDAD-MAX-30", confidence=0.5)],
            decision=decision(),
            operator_events=[],
        )

        self.assertEqual(enriched[0]["track_id"], 1)
        rules = {flag["rule"] for flag in flags}
        self.assertIn("low_confidence_band", rules)
        self.assertIn("track_class_change", rules)

    def test_short_lived_detection_is_flagged_when_it_disappears(self):
        detector = analyzer()
        detector.evaluate(
            frame_seq=1,
            frame_shape=(240, 320, 3),
            predictions=[prediction()],
            decision=decision(),
            operator_events=[],
        )

        _enriched, flags = detector.evaluate(
            frame_seq=2,
            frame_shape=(240, 320, 3),
            predictions=[],
            decision=decision(),
            operator_events=[],
        )

        self.assertIn("short_lived_detection", {flag["rule"] for flag in flags})

    def test_ambiguous_decision_and_operator_override_are_flagged(self):
        _enriched, flags = analyzer().evaluate(
            frame_seq=1,
            frame_shape=(240, 320, 3),
            predictions=[],
            decision=decision(action="ambiguous", state="ambiguous", reason="conflict"),
            operator_events=[{"seq": 1, "type": "manual_override", "reason": "stop"}],
        )

        rules = {flag["rule"] for flag in flags}
        self.assertIn("ambiguous_decision", rules)
        self.assertIn("operator_override", rules)


class SessionRecorderTest(unittest.TestCase):
    def test_recorder_writes_manifest_labels_and_critical_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SessionRecorder(
                Path(tmp),
                autostart=False,
                save_images=True,
                min_interval_sec=0.0,
                jpeg_quality=80,
                save_video=False,
                video_fps=5.0,
                save_critical_images=True,
            )
            status = recorder.start()
            self.assertTrue(status["enabled"])

            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            recorder.record(
                frame=frame,
                frame_seq=1,
                predictions=[prediction(confidence=0.45)],
                inference_payload={"predictions": []},
                decision=decision(),
                inference_latency_ms=12,
                inference_backend={"api_url": "http://test"},
                control={"mode": "autonomous"},
                operator_events=[],
            )

            session_dir = Path(recorder.snapshot()["session_dir"])
            self.assertEqual(recorder.snapshot()["records"], 1)
            self.assertEqual(recorder.snapshot()["critical_records"], 1)
            self.assertTrue((session_dir / "manifest.jsonl").exists())
            self.assertTrue((session_dir / "labels.jsonl").exists())
            self.assertTrue((session_dir / "critical.jsonl").exists())
            self.assertTrue((session_dir / "images/frame_00000001.jpg").exists())
            self.assertTrue((session_dir / "critical/frame_00000001.jpg").exists())


if __name__ == "__main__":
    unittest.main()
