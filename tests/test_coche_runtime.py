from __future__ import annotations

import unittest

from coche import NEUTRAL_STEERING, NEUTRAL_THROTTLE, RuntimeState


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


if __name__ == "__main__":
    unittest.main()
