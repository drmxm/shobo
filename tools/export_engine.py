#!/usr/bin/env python3
import argparse, json, os, shutil
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO  # ensures cv2/torch are imported correctly

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path or name (e.g., yolov8n.pt)")
    ap.add_argument("imgsz", nargs="?", type=int, default=640)
    ap.add_argument("--outdir", default="ros2_ws", help="Host-visible output dir (mounted)")
    ap.add_argument("--engine-name", default=None, help="Filename for .engine (default: <modelstem>.engine)")
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--device", default=0, help="GPU device id (default 0)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--workspace", type=int, default=4, help="TRT workspace (GB)")
    args = ap.parse_args()

    work = Path("/work")
    outdir = (work / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = args.model
    model_stem = Path(model_path).stem
    engine_name = args.engine_name or f"{model_stem}.engine"
    engine_dst = (outdir / engine_name).resolve()

    print(f"[export] model={model_path} imgsz={args.imgsz} → {engine_dst}")

    # Export with Ultralytics → TensorRT
    m = YOLO(model_path)
    export_path = m.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
        opset=args.opset,
        workspace=args.workspace,
        dynamic=False,
        simplify=True,
        verbose=True,
        project=str(outdir),  # place runs under outdir
        name="trt_export",
        exist_ok=True,
    )

    # Ultralytics returns path(s). Normalize to a single .engine file path.
    if isinstance(export_path, (list, tuple)):
        export_path = export_path[0]
    export_path = Path(export_path).resolve()

    # If Ultralytics wrote to a different path (e.g., outdir/trt_export/weights/*.engine) → copy to desired name
    if export_path != engine_dst:
        engine_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(export_path, engine_dst)

    # Minimal metadata to pair with engine
    meta = {
        "engine": str(engine_dst.relative_to(work)),
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "imgsz": args.imgsz,
        "half": bool(args.half),
        "device": args.device,
        "opset": args.opset,
        "workspace_gb": args.workspace,
        "source_model": model_path,
    }
    meta_path = outdir / f"{model_stem}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[ok] engine written: {engine_dst.relative_to(work)}")
    print(f"[ok] metadata: {meta_path.relative_to(work)}")

if __name__ == "__main__":
    # Make sure config dir exists to avoid Ultralytics warning
    os.makedirs("/work/.ultralytics", exist_ok=True)
    main()
