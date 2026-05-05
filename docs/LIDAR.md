# TP2 LiDAR Runtime Notes

## Current Integration

LiDAR is integrated as an EPC-owned safety input for `servicios/coche.py`.

- The car sends scans over the existing UDP path to EPC.
- Preferred packet discriminator: `L`.
- Payload formats accepted by EPC:
  - `pickle` payload with `ranges` plus `angle_min` and `angle_increment`.
  - raw JSON with the same LaserScan-like fields.
  - Cartesian `points` as `[x, y, z, intensity]` lists or objects.
  - LiDAR nested in telemetry `D` under `lidar`, `lidar_scan`, `scan`, `ranges`, or `points`.
- EPC publishes LiDAR status and a capped point list in `/status.json`.
- The web UI can switch the main stage between camera and LiDAR reconstruction.

## Safety Behavior

The current safety layer is deterministic and runs before the final UDP control packet is sent.

- `TP2_LIDAR_STOP_DISTANCE_M` default `0.42`: frontal obstacle forces `lidar-stop` and neutral throttle.
- `TP2_LIDAR_SLOW_DISTANCE_M` default `0.85`: frontal obstacle caps throttle to `TP2_LIDAR_SLOW_THROTTLE`.
- `TP2_LIDAR_CAUTION_DISTANCE_M` default `1.35`: frontal obstacle can add a bounded steering correction.
- `TP2_LIDAR_STALE_SEC` default `0.75`: stale LiDAR does not override camera/autonomous decisions.

This keeps EPC as the control hub. Jetson remains inference-only if a learned LiDAR model is added later.

## Candidate Learned Models

Recommended next step for this lab: start with PointPillars on Jetson/TensorRT after collecting synchronized local LiDAR frames, then consider fusion models only after the data contract is stable.

- PointPillars: good first learned LiDAR detector because it is designed for fast 3D detection from point clouds. The original paper reports 62 Hz for the full pipeline and 105 Hz for a faster variant on KITTI-style data: https://arxiv.org/abs/1812.05784
- NVIDIA TAO PointPillars: practical deployment path for NVIDIA hardware and TensorRT-oriented workflows: https://docs.nvidia.com/tao/tao-toolkit/text/cv_finetuning/pytorch/point_cloud/pointpillars.html
- CUDA-PointPillars: NVIDIA Jetson-focused reference for CUDA/TensorRT deployment: https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-with-cuda-pointpillars/
- CenterPoint: stronger LiDAR detection/tracking candidate once runtime budget and dataset are better understood: https://arxiv.org/abs/2006.11275
- BEVFusion: camera-LiDAR fusion candidate for a later phase because it needs synchronized camera/LiDAR calibration and dataset work: https://arxiv.org/abs/2205.13542 and https://github.com/mit-han-lab/bevfusion
- Autoware obstacle collision checking: useful architectural reference for trajectory/point-cloud collision checks, not a direct drop-in for this script-first lab: https://autowarefoundation.github.io/autoware_universe/pr-10077/control/autoware_obstacle_collision_checker/

## Validation Checklist

1. Start LTE and `tp2-car-control.service`.
2. Confirm `/status.json` receives LiDAR frames and reports point count.
3. Switch the web stage to LiDAR and confirm point reconstruction is visible.
4. In manual mode, confirm LiDAR data updates without changing control.
5. In autonomous mode, place an obstacle inside the stop distance and confirm `lidar-stop`.
6. Record validation evidence under `docs/logs/validations/`.
