from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import skimage.io
import skimage.transform
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import sys


BASE_DIR = Path(__file__).resolve().parent
RETINANET_LIB = BASE_DIR / "Monk_Object_Detection" / "5_pytorch_retinanet" / "lib"
MODEL_PATH = BASE_DIR / "final_model.pt"
CLASSES_PATH = BASE_DIR / "Waste_Dataset" / "annotations" / "classes.txt"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "Inference"
WEB_DIR = BASE_DIR / "web"

MIN_SIDE = 608
MAX_SIDE = 1024
MEAN = np.array([[[0.485, 0.456, 0.406]]])
STD = np.array([[[0.229, 0.224, 0.225]]])
COLORS = {
    "dry_waste": (0, 165, 255),
    "wet_waste": (0, 180, 0),
}
EXPECTED_CLASSES = ["dry_waste", "wet_waste"]

if str(RETINANET_LIB) not in sys.path:
    sys.path.append(str(RETINANET_LIB))

app = FastAPI(
    title="Waste Segregation API",
    description="Upload a waste image and get dry/wet waste detections.",
    version="1.0.0",
)
WEB_DIR.mkdir(exist_ok=True)
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

model = None
class_list = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classes():
    if not CLASSES_PATH.exists():
        return EXPECTED_CLASSES.copy()

    classes = []
    for line in CLASSES_PATH.read_text().splitlines():
        label = line.strip()
        if label:
            classes.append(label)

    if classes != EXPECTED_CLASSES:
        raise RuntimeError(
            f"{CLASSES_PATH} must contain exactly: {', '.join(EXPECTED_CLASSES)}. "
            f"Found: {', '.join(classes) or 'no classes'}"
        )

    return classes


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    try:
        loaded_model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    except TypeError:
        loaded_model = torch.load(MODEL_PATH, map_location=device)

    loaded_model = loaded_model.to(device)
    loaded_model.eval()
    return loaded_model


@app.on_event("startup")
def startup():
    global model, class_list
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    class_list = load_classes()
    model = load_model()


def prepare_image(image_path):
    image_path = Path(image_path)
    image = skimage.io.imread(str(image_path))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]

    image = image.astype(np.float32) / 255.0
    image = (image.astype(np.float32) - MEAN) / STD

    rows, cols, _ = image.shape
    smallest_side = min(rows, cols)
    scale = MIN_SIDE / smallest_side

    largest_side = max(rows, cols)
    if largest_side * scale > MAX_SIDE:
        scale = MAX_SIDE / largest_side

    image = skimage.transform.resize(
        image,
        (int(round(rows * scale)), int(round(cols * scale))),
        preserve_range=True,
    )

    rows, cols, channels = image.shape
    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    padded = np.zeros((rows + pad_w, cols + pad_h, channels)).astype(np.float32)
    padded[:rows, :cols, :] = image.astype(np.float32)

    tensor = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0)
    return tensor.to(device), scale


def run_prediction(image_path, threshold):
    image_path = Path(image_path)
    image_tensor, scale = prepare_image(image_path)

    with torch.no_grad():
        scores, labels, boxes = model(image_tensor)

    boxes = boxes / scale
    detections = []
    output_image = cv2.imread(str(image_path))
    image_height, image_width = output_image.shape[:2]

    for score, label, box in zip(scores, labels, boxes):
        confidence = float(score)
        if confidence < threshold:
            break

        class_id = int(label)
        if class_id < 0 or class_id >= len(class_list):
            continue

        class_name = class_list[class_id]
        x1, y1, x2, y2 = [int(value) for value in box.tolist()]
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        detections.append(
            {
                "class_id": class_id,
                "detected_object": class_name,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "segregation_confidence": round(confidence, 4),
                "segregation_method": "model detection",
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "width": max(x2 - x1, 0),
                "height": max(y2 - y1, 0),
                "area_pixels": max(x2 - x1, 0) * max(y2 - y1, 0),
            }
        )

        color = COLORS.get(class_name, (255, 0, 0))
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        label_text = f"{class_name}: {confidence:.2f}"
        cv2.putText(
            output_image,
            label_text,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    if detections:
        best_detection = detections[0]
        image_waste_type = best_detection["class_name"]
        image_waste_confidence = best_detection["confidence"]
        image_waste_reason = "model detection"
    else:
        image_waste_type = "no_detection"
        image_waste_confidence = 0
        image_waste_reason = "model detection"

    output_name = f"{image_path.stem}_prediction.jpg"
    output_path = OUTPUT_DIR / output_name
    cv2.imwrite(str(output_path), output_image)
    return (
        detections,
        output_name,
        {"width": image_width, "height": image_height},
        {
            "class_name": image_waste_type,
            "confidence": image_waste_confidence,
            "method": image_waste_reason,
        },
    )


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web page not found")
    return index_path.read_text(encoding="utf-8")


@app.get("/api")
def api_info():
    return {
        "message": "Waste Segregation API is running",
        "model": MODEL_PATH.name,
        "device": str(device),
        "classes": class_list,
        "docs": "/docs",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.4):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="Upload an image file")

    image_name = f"{uuid4().hex}{suffix}"
    image_path = UPLOAD_DIR / image_name
    image_path.write_bytes(await file.read())

    detections, output_name, image, waste_type = run_prediction(image_path, threshold)
    best_detection = detections[0] if detections else None
    class_counts = {}
    for detection in detections:
        class_name = detection["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return {
        "filename": file.filename,
        "threshold": threshold,
        "image": image,
        "waste_type": waste_type,
        "detections": detections,
        "detection_count": len(detections),
        "class_counts": class_counts,
        "best_detection": best_detection,
        "result_image": f"/result/{output_name}",
    }


@app.get("/result/{filename}")
def result_image(filename: str):
    safe_name = Path(filename).name
    output_path = OUTPUT_DIR / safe_name
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Result image not found")
    return FileResponse(output_path, media_type="image/jpeg")
