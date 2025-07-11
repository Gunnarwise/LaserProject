# yolo_tracker.py

from ultralytics import YOLO
import cv2

# Load YOLOv8n model once (fast + light version)
model = YOLO("yolov8n.pt")

def detect_objects(frame, target_classes=None):
    """
    Runs YOLO object detection on the input frame.
    Returns list of dicts: label, confidence, bbox, center
    Optionally filter by target_classes (list of class names).
    """
    results = model(frame, stream=True)
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])

            if target_classes and label not in target_classes:
                continue

            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2)
            })

    return detections