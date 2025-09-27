# Shobo Perception Stack

End-to-end perception stack for Jetson-based robots. It streams RGB (UVC) and IR (CSI) cameras into TensorRT-accelerated YOLOv8 detectors, then publishes annotated images and detections for downstream consumers (e.g., Foxglove Studio, autonomy nodes).

## System Diagram
- **shobo_sensors** – Captures RGB over UVC (`/sensors/rgb/...`) and IR over CSI (`/sensors/ir/...`).
- **shobo_detectors** – Runs YOLOv8 TensorRT engines for each stream, publishes `/perception/<stream>/detections` and optional annotated images.
- **infra** (optional) – Foxglove bridge, rosbridge, etc. for visualization and telemetry.

## Quick Start (Jetson Host)
1. Allow X forwarding once per shell session:
   ```bash
   xhost +local:root
   ```
2. Build & start the perception stack:
   ```bash
   docker compose build perception
   docker compose up perception
   ```
3. Re-launch without rebuilding:
   ```bash
   docker compose up --no-build perception
   ```
4. Stop and rebuild the detection container when the engine changes:
   ```bash
   docker compose stop shobo-perception || true
   docker compose rm -f -s -v perception
   docker compose build --no-cache perception
   docker builder prune -af
   xhost +local:root
   docker compose up --no-build perception
   ```

To run the detector container by hand (outside compose) the TensorRT engine is
auto-generated on first launch from the bundled ONNX (`/work/ros2_ws/yolov8n.onnx`),
so no volume mount is required. The initial run will spend ~1–2 minutes building
the engine and cache:

```bash
xhost +local:root  # once per shell if you need X11/GL
docker run --rm -it --net=host --runtime nvidia --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /dev:/dev \
  shobo-perception
```

docker run -it --net=host --runtime nvidia --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /dev:/dev \
  shobo-perception
docker run --rm --entrypoint /usr/local/bin/ensure_trt_engine.sh shobo-perception --check

what I vedone
docker compose rm -f -s -v perception
CUDA_MODULE_LOADING=LAZY docker compose up perception
docker compose exec perception bash
### Useful Environment Variables
- `RGB_DEV` – Override the UVC device (auto-detects `/dev/video0`/`/dev/video1`).
- `IR_SENSOR_ID` – Pick CSI sensor index (default `0`).
- `YOLO_ENGINE` – Path to the TensorRT engine inside the container (default `/work/ros2_ws/models/yolov8n.engine`).
- `YOLO_ONNX` – Override the ONNX path used when regenerating the engine (default `/work/ros2_ws/yolov8n.onnx`).
- `PUBLISH_ANNOTATED` – Set to `0/false` to disable annotated image publication.

## Runtime Topics
| Stream | Image topic | Camera info | Annotated | Detections |
|--------|-------------|-------------|-----------|------------|
| RGB    | `/sensors/rgb/image_raw` | `/sensors/rgb/camera_info` | `/perception/rgb/annotated` | `/perception/rgb/detections` |
| IR     | `/sensors/ir/image_raw`  | `/sensors/ir/camera_info`  | `/perception/ir/annotated`  | `/perception/ir/detections`  |

Detections are `vision_msgs/Detection2DArray` with COCO labels loaded from `config/detectors.yaml`.

## Building the TensorRT Engine
The container now self-manages the TensorRT plan: on start it checks
`/work/ros2_ws/models/yolov8n.engine`, and if missing or stale compared to the
ONNX file it runs `trtexec` with the Jetson GPU to regenerate it. This guarantees
version compatibility with the deployed JetPack. The engine is no longer bind-mounted
from the host in `docker-compose.yml`, so TensorRT is always free to emit a fresh
plan that matches the device you launch on. The bundled ONNX already encodes the
static `1x3x640x640` input shape, so the builder relies on that shape instead of
forcing an optimization profile. If you export a dynamic variant for multi-shape
support, pass your own `--minShapes/--optShapes/--maxShapes` flags when rebuilding the engine.

If you prefer to bake a custom model ahead of time, you can still export ONNX or
engines using the tooling under `tools/` and drop the files into `ros2_ws/` before
building the image. Set `YOLO_ONNX`/`YOLO_ENGINE` at runtime to point at the
appropriate files.

## Detector Node Configuration
`config/detectors.yaml` ships defaults for both detectors:
- Per-stream topics and COCO class labels.
- Confidence/IoU thresholds, TensorRT bindings, CUDA pre-processing toggle.
- Shared engine path overrideable via `YOLO_ENGINE` environment variable.

Each detector instance can be reconfigured via ROS parameters or launch-time overrides. Notable parameters:
- `input_topic`, `annotated_topic`, `detections_topic`
- `engine_path`, `input_binding`, `output_binding`
- `conf_th`, `iou_th`, `max_detections`
- `use_cuda_preproc` (GPU letterbox + normalization)
- `publish_annotated`
- Ensure the JetPack image provides the TensorRT companion libraries (`libnvdla-compiler`, `libnvdla-runtime`, `cuda-cudla`). The Dockerfile installs them automatically so the detector links cleanly on-device.

## Development Notes
- CUDA preprocessing now performs bilinear letterboxing directly on the GPU. Set `use_cuda_preproc:=false` via parameters if you need a CPU fallback for debugging.
- Launch file spins two detector nodes: `trt_detector_rgb` and `trt_detector_ir`.
- `shobo_detectors` automatically falls back to a stub if TensorRT headers/libs are missing at build time.


# List topics quickly
ros2 topic list

# Camera FPS
ros2 topic hz /sensors/rgb/image_raw
ros2 topic hz /sensors/ir/image_raw

# Detector output FPS
ros2 topic hz /perception/rgb/detections
ros2 topic hz /perception/ir/detections

# Peek first few detections
ros2 topic echo /perception/rgb/detections -n 5 --qos-reliability best_effort
ros2 topic echo /perception/ir/detections  -n 5 --qos-reliability best_effort


## Improvement Plan
1. Add automated regression tests (bag replay) covering RGB + IR detection throughput and accuracy.
2. Implement runtime health reporting (camera frame age, detector inference time) and expose on diagnostics topics.
3. Optimize IR preprocessing for lower light (gamma/LUT) prior to inference.
4. Integrate model auto-reload when `yolov8n.json` changes to reduce downtime during updates.
5. Document Foxglove/infra stack with sample dashboards and alerting hooks.

## Troubleshooting
- **No detections:** Confirm `/work/ros2_ws/models/yolov8n.engine` exists inside the container and matches binding names (`images`, `output0`).
- **Camera timeouts:** Check device permissions (`/dev/video*` mapped) and that `RGB_DEV` / `IR_SENSOR_ID` are correct.
- **TensorRT errors at launch:** Rebuild inside matching JetPack/L4T versions. Engines are not portable across JetPack releases.
- **High latency:** Disable annotated publishing (`PUBLISH_ANNOTATED=0`) or lower image size (regenerate engine with `imgsz=512`).

---
For more context about the supporting services (Foxglove, telemetry) see `infra/README.md`.
