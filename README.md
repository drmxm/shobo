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
   docker compose stop perception || true
   docker compose rm -f -s -v perception
   docker compose build --no-cache perception
   docker builder prune -af
   xhost +local:root
   docker compose up --no-build perception
   docker build --no-cache --progress=plain   --build-arg L4T_TAG=r36.4.0   -t shobo-perception .
   docker run --rm -it --net=host --runtime nvidia --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /dev:/dev shobo-perception
  
   ```

   ```bash
   xhost +local:root  # if you use RViz or anything X11
docker run --rm -it --net=host --runtime nvidia --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /dev:/dev \
  shobo-perception
   ```

### Useful Environment Variables
- `RGB_DEV` – Override the UVC device (default `/dev/video1`).
- `IR_SENSOR_ID` – Pick CSI sensor index (default `0`).
- `YOLO_ENGINE` – Path to the TensorRT engine inside the container (default `/work/ros2_ws/yolov8n.engine`).
- `PUBLISH_ANNOTATED` – Set to `0/false` to disable annotated image publication.

## Runtime Topics
| Stream | Image topic | Camera info | Annotated | Detections |
|--------|-------------|-------------|-----------|------------|
| RGB    | `/sensors/rgb/image_raw` | `/sensors/rgb/camera_info` | `/perception/rgb/annotated` | `/perception/rgb/detections` |
| IR     | `/sensors/ir/image_raw`  | `/sensors/ir/camera_info`  | `/perception/ir/annotated`  | `/perception/ir/detections`  |

Detections are `vision_msgs/Detection2DArray` with COCO labels loaded from `config/detectors.yaml`.

## Building the TensorRT Engine
The repo carries a reference `yolov8n.engine`, but you can regenerate it for other models or image sizes.

### 1. Export YOLOv8 to TensorRT (recommended)
Use the helper script which builds the engine container and runs Ultralytics export inside it:
```bash
./tools/build_engine.sh yolov8n.pt 640
```
- Uses `docker compose` profile `engine` (`tools/Dockerfile.engine`).
- Outputs the engine and metadata into `ros2_ws/` (e.g., `ros2_ws/yolov8n.engine`, `ros2_ws/yolov8n.json`).
- Tweak arguments (e.g., `yolov8s.pt`, custom image size) as needed.

### 2. Manual Ultralytics export inside the container
```bash
docker compose run --rm engine \
  python3 tools/export_engine.py yolov8n.pt 640 \
  --outdir ros2_ws --workspace 6 --half
```
- `--half` enables FP16 (default `True`).
- `--workspace` sets TensorRT workspace size (GB).
- The script writes `ros2_ws/trt_export/weights/*.engine` then copies to `ros2_ws/<model>.engine`.

### 3. Produce an intermediate ONNX (optional)
If you need to inspect or edit the ONNX before TensorRT conversion:
```bash
docker compose run --rm engine \
  yolo export model=yolov8n.pt format=onnx imgsz=640 simplify=True
```
The ONNX will be stored under `/work/runs/export/onnx/` inside the container; copy or mount as required. You can then feed it into TensorRT using `trtexec` or custom tooling.

### 4. Validating the Engine
1. Copy the new `.engine` + `.json` into `ros2_ws/` (mounted read-only into the perception container).
2. Restart the `perception` service.
3. Monitor topics:
   ```bash
   ros2 topic echo /perception/rgb/detections
   ros2 topic echo /perception/ir/detections
   ```
4. Optionally view overlays in Foxglove (`ws://<jetson-ip>:8765`).

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

## Improvement Plan
1. Add automated regression tests (bag replay) covering RGB + IR detection throughput and accuracy.
2. Implement runtime health reporting (camera frame age, detector inference time) and expose on diagnostics topics.
3. Optimize IR preprocessing for lower light (gamma/LUT) prior to inference.
4. Integrate model auto-reload when `yolov8n.json` changes to reduce downtime during updates.
5. Document Foxglove/infra stack with sample dashboards and alerting hooks.

## Troubleshooting
- **No detections:** Confirm `/work/ros2_ws/yolov8n.engine` exists inside the container and matches binding names (`images`, `output0`).
- **Camera timeouts:** Check device permissions (`/dev/video*` mapped) and that `RGB_DEV` / `IR_SENSOR_ID` are correct.
- **TensorRT errors at launch:** Rebuild inside matching JetPack/L4T versions. Engines are not portable across JetPack releases.
- **High latency:** Disable annotated publishing (`PUBLISH_ANNOTATED=0`) or lower image size (regenerate engine with `imgsz=512`).

---
For more context about the supporting services (Foxglove, telemetry) see `infra/README.md`.
