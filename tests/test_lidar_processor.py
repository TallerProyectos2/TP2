from __future__ import annotations

import json
import math
import unittest

from lidar_processor import (
    LidarConfig,
    analyze_lidar_scan,
    normalize_lidar_payload,
)


class LidarProcessorTest(unittest.TestCase):
    def test_ranges_payload_builds_forward_points(self):
        scan = normalize_lidar_payload(
            {
                "ranges": [1.0, 2.0, 3.0],
                "angle_min": -math.radians(10),
                "angle_increment": math.radians(10),
            },
            received_at=10.0,
        )

        self.assertEqual(len(scan.points), 3)
        self.assertAlmostEqual(scan.points[1].x, 0.0, places=3)
        self.assertAlmostEqual(scan.points[1].y, 2.0, places=3)

    def test_json_payload_is_accepted(self):
        payload = json.dumps(
            {
                "source": "unit-test",
                "ranges": [0.3],
                "angle_min": 0.0,
                "angle_increment": 0.0,
            }
        ).encode("utf-8")

        scan = normalize_lidar_payload(payload, received_at=10.0)

        self.assertEqual(scan.source, "unit-test")
        self.assertEqual(len(scan.points), 1)

    def test_front_obstacle_stops(self):
        config = LidarConfig(stop_distance_m=0.4, slow_distance_m=0.8)
        scan = normalize_lidar_payload(
            {"points": [{"x": 0.0, "y": 0.32, "z": 0.0}]},
            config=config,
            received_at=10.0,
        )

        safety = analyze_lidar_scan(scan, config=config, now=10.1)

        self.assertEqual(safety.status, "stop")
        self.assertEqual(safety.throttle_limit, 0.0)

    def test_side_obstacle_does_not_stop_front_path(self):
        config = LidarConfig(front_angle_deg=25.0)
        scan = normalize_lidar_payload(
            {"points": [{"x": 0.7, "y": 0.2, "z": 0.0}]},
            config=config,
            received_at=10.0,
        )

        safety = analyze_lidar_scan(scan, config=config, now=10.1)

        self.assertEqual(safety.status, "clear")

    def test_empty_scan_is_not_reported_as_clear(self):
        config = LidarConfig()
        scan = normalize_lidar_payload({"points": []}, config=config, received_at=10.0)

        safety = analyze_lidar_scan(scan, config=config, now=10.1)

        self.assertEqual(safety.status, "empty")
        self.assertFalse(safety.clear)


if __name__ == "__main__":
    unittest.main()
