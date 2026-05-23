"""
High-performance test script for RGB & Thermal YOLOv8 inference.
Usage: python src/detection/test_detector.py --image <path_to_image> --weights yolov8n.pt
"""
import sys
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
import torch

# Monkeypatch torch.load to default weights_only=False for PyTorch 2.6+ compatibility with Ultralytics YOLO
try:
    orig_load = torch.load
    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)
    torch.load = patched_load
except Exception:
    pass

def run_inference(image_path, weights_path, output_path, apply_colormap = False):
    path = Path(image_path)
    if not path.exists():
        print(f" Error: Image not found at {image_path}")
        sys.exit(1)

    # 1. Load image (supports grayscale/thermal and RGB)
    # cv2.IMREAD_UNCHANGED keeps original channels
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(" Error: Could not decode image.")
        sys.exit(1)

    print(f" Loaded image of shape: {img.shape}")

    # 2. Preprocess Thermal/Grayscale/BGRA to 3-Channel BGR for YOLOv8
    if len(img.shape) == 3 and img.shape[2] == 4:
        print(" 4-channel (RGBA/BGRA) image detected. Stripping alpha channel...")
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) == 2 or img.shape[2] == 1:
        print(" Grayscale/Thermal image detected. Converting to 3-channel BGR...")
        if apply_colormap:
            img_bgr = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    # 3. Load YOLO model
    print(f" Loading YOLO model with weights: {weights_path}")
    model = YOLO(weights_path)

    # 4. Run inference
    results = model(img_bgr, conf=0.25, verbose=False)
    result = results[0]

    # 5. Parse and print results
    boxes = result.boxes
    print(f"\n Detections: {len(boxes)}")
    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        cls_name = model.names[cls_id]
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        print(f"  [{i+1}] {cls_name.upper()} | Conf: {conf:.2f} | Bbox: {[int(x) for x in xyxy]}")

    # 6. Save annotated image
    annotated_img = result.plot()
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_p), annotated_img)
    print(f" Success! Saved annotated result to: {out_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGB/Thermal YOLOv8 Tester")
    parser.add_argument("--image", required=True, help="Path to input image (RGB or Thermal)")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to YOLOv8 weights (default: yolov8n.pt)")
    parser.add_argument("--output", default="data/processed/test_output.jpg", help="Path to save output")
    parser.add_argument("--colormap", action="store_true", help="Apply false JET colormap to thermal grayscale input")
    args = parser.parse_args()

    run_inference(args.image, args.weights, args.output, args.colormap)
