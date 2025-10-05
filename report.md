# Shobo Perception Platform – Executive Overview

## Executive Summary
- End-to-end perception stack already running on **Jetson Orin (GPU device 0)** with NVIDIA JetPack 6.2, delivering TensorRT-accelerated YOLOv8 detections for RGB and IR streams.
- Containerized deployment (`shobo-perception`, `shobo-infra`) brings cameras, inference, visualization, and ROS 2 middleware under one repeatable workflow.
- Current maturity: production-ready vision baseline with known items to harden (camera bring-up robustness, regression automation). The platform is a strong foundation for layering additional sensors and analytics.

## Current Feature Set (Technically Meaningful & Customer Facing)
- **Dual-Spectrum Vision**: UVC RGB and CSI IR capture via `shobo_sensors`; synchronized annotated feeds for downstream UX (e.g., Foxglove dashboards, teleoperations).
- **Real-Time Object Detection**: YOLOv8n converted to TensorRT, steady 640×640 inference on Orin with FP16 precision; configurable confidence/IoU thresholds per stream.
- **Self-Managing Models**: `ensure_trt_engine.sh` checks ONNX freshness and rebuilds engines in-container, ensuring hardware-aligned plans without manual intervention.
- **ROS-Native Interfaces**: Standardized topics (`/sensors/*`, `/perception/*`) simplify integration with autonomy stacks, SLAM, or logging nodes.
- **Operator Tooling**: Docker Compose orchestration, Foxglove-ready infra container, and documented bring-up steps reduce deployment friction for field teams.

## Architecture at a Glance
| Layer | Components | Responsibilities | Benefits |
|-------|------------|-----------------|----------|
| **Sensor Layer** | `shobo_sensors` (uvc_cam_node, csi_cam_node) | Camera discovery, calibration, frame publication | Modular driver pattern enables new modalities without touching inference layer |
| **Perception Layer** | `shobo_detectors` (TensorRT YOLOv8) | Preprocessing, inference, Detection2D output, optional annotation | GPU-optimized inference, ROS-native messaging for easy downstream consumption |
| **Infrastructure Layer** | `shobo-infra` services, Foxglove bridge | Visualization, telemetry, health | Gives operations team situational awareness; allows remote tuning and demos |
| **Deployment & Ops** | Dockerfile + Compose, TRT engine manager | Repeatable builds, GPU provisioning, runtime health logs | Fast provisioning on new Jetsons; fewer manual steps for technicians |

## GPU & Deployment Footprint
- **Hardware**: Jetson Orin (8 SMs, 7.6 GB VRAM). Logs confirm TensorRT selects device `GPU-2998b673…` (compute capability 8.7).
- **Runtime**: Dockerized ROS 2 Humble workspace, TensorRT 10.3.0, CUDA 11.8. All inference runs on GPU; CPU handles fallback preprocessing when CUDA kernels disabled.
- **Performance Envelope**: 30 FPS camera ingest targets; CPU letterboxing currently used (safe, ~3–4 ms overhead). CUDA preproc available once validated.

## Near-Term Roadmap (0–3 Months)
- **0–4 Weeks**
  - Harden camera initialization (headless EGL or automated `xhost`), add frame-age watchdogs.
  - Create regression suite (RGB/IR rosbag replay + YOLO decode unit tests) and wire into CI.
  - Reorganize assets (`ros2_ws/models/` for weights, remove stray `.deb` files) and expand README for one-click bring-up.
- **1–2 Months**
  - Integrate thermal camera driver; tune detection thresholds per modality.
  - Prototype depth/LiDAR ingestion and publish `/sensors/depth/...` topics for fusion experiments.
  - Re-enable GPU preprocessing with coverage tests; benchmark FP16 vs INT8 engines.
- **2–3 Months**
  - Add diagnostics/metrics publisher, feed into Foxglove + Prometheus.
  - Implement detector watchdog (auto restart on sustained zero detections) and lifecycle transitions.
  - Prepare field pilot package with demo scripts (`Demo-shobo-v1.0.0.webm`) and updated Foxglove workspace.

## Planned Sensor Integrations & Value
- **Thermal (FLIR/SIP)**: Enables low-light person detection and temperature anomalies. Requires GStreamer pipeline, calibration, and possibly dedicated model.
- **Depth / LiDAR**: Unlocks 3D localization of detections, safer navigation, and occupancy grid generation. Pairs well with existing detection nodes for tracking.
- **IMU / Radar** (future): Improves motion estimation, supports tracking filter, provides redundancy in adverse visibility.
- **Acoustic or Environmental Sensors** (stretch): Adds condition monitoring for industrial scenarios—tie into the same ROS diagnostics framework.

## Differentiators & Benefits
- **Modular yet Integrated**: Clear layering (sensors → inference → infra) allows rapid sensor additions without destabilizing core perception.
- **GPU-Optimized Stack**: Uses native TensorRT engines aligned to Jetson hardware, offering competitive FPS and power efficiency versus CPU or generic runtimes.
- **Operator-Friendly Deployment**: Docker Compose recipes, auto engine rebuilds, and annotated feeds reduce on-site setup time—critical for sales demos and pilots.
- **ROS 2 First-Class Citizen**: Immediate compatibility with existing autonomy stacks and off-the-shelf tools (Nav2, SLAM, Foxglove), accelerating integration timelines.
- **Scalable Roadmap**: Plan covers diagnostics, fusion, and performance improvements, showing commitment to reliability and extensibility—key messages for prospects.

## Risks & Mitigations
- **Camera EGL Dependency**: Current CSI/UVC flow requires host X11 access (`xhost`). Mitigation: move to headless EGLStream or document automated permission steps.
- **Engine Rebuild Duration**: First boot compilation (~10 min) can slow demos. Mitigation: cache architecture-specific engines or pre-bake with release images.
- **Sensor Expansion Unknowns**: Need final SKU list to prioritize drivers. Mitigation: maintain modular sensor interface and allocate discovery sprints per new modality.

## Why This Platform Matters
- Provides a **turnkey perception baseline** that demo teams can showcase today while engineering expands capabilities.
- Creates a **single integration point** for future sensors, cutting development time when new hardware arrives.
- Demonstrates a **balanced roadmap**: shipping value immediately (dual-spectrum detections) while outlining concrete steps toward sensor fusion and advanced analytics.
- Aligns with customer pain points—fast deployment, reliable detection in varied lighting, and extensibility—making it a strong story for sales while staying honest about pending work.

