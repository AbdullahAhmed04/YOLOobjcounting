"""
AI Object Counting - local demo app (no Google Drive, no Colab, no ngrok).
Run:  python app.py      then open http://127.0.0.1:8000
Optional settings (environment variables):
  MODEL=yolov8x.pt   YOLO weights (default: yolov8x on GPU, yolov8s on CPU)
  TTA=1              test-time augmentation (default: on for GPU, off for CPU)
  FACE_FALLBACK=0    turn off the OpenCV face-cascade fallback for "person"
  PORT=8000          port to use
"""
import io, base64, csv, os, threading, webbrowser
from ultralytics import YOLO
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
ON_CLOUD = bool(os.environ.get("ON_CLOUD") or os.environ.get("K_SERVICE"))   # set in the Dockerfile (works on Azure, Cloud Run, Railway)
INFER_LOCK = threading.Lock()                  # one detection at a time (the model is shared)
LOG_DIR = os.environ.get("DATA_DIR") or ("/tmp/data" if ON_CLOUD else os.path.join(BASE, "data"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "results_log.csv")
EVAL_PATH = os.path.join(LOG_DIR, "evaluation_table.csv")
LOG_LOCK = threading.Lock()

HAS_GPU = torch.cuda.is_available()
MODEL_NAME = os.environ.get("MODEL", "yolov8x.pt" if HAS_GPU else "yolov8s.pt")
USE_TTA = os.environ.get("TTA", "1" if HAS_GPU else "0") == "1"
FACE_FALLBACK = os.environ.get("FACE_FALLBACK", "1") == "1"

model = YOLO(MODEL_NAME)
ALL_CLASSES = list(model.names.values())
NAME_TO_ID = {v: k for k, v in model.names.items()}
print(f"Model: {MODEL_NAME} | GPU: {HAS_GPU} | TTA: {USE_TTA} | face fallback: {FACE_FALLBACK}")
print(f"Results are saved in: {LOG_DIR}")

from datetime import datetime
from collections import Counter
import cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_MB = 10
FIELDS = ["input_filename", "timestamp", "confidence", "image_size",
          "total_count", "per_class_counts", "actual_count"]

app = FastAPI(title="AI Object Counting API")

# Load pre-trained face cascade for fallback close-up detection
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise RuntimeError("cascade file not found")
except Exception as e:
    print("Face fallback disabled:", e)
    face_cascade = None
    FACE_FALLBACK = False

# ---------- helpers (callers hold LOG_LOCK) ----------
def read_rows():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["actual_count"] = (r.get("actual_count") or "").strip()
    return rows

def write_rows(rows):
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def evaluated(rows):
    out = []
    for i, r in enumerate(rows):
        a = r.get("actual_count", "")
        if a == "":
            continue
        try:
            actual, pred = int(a), int(r["total_count"])
        except (ValueError, KeyError):
            continue
        err = pred - actual
        acc = (1.0 if pred == 0 else 0.0) if actual == 0 else max(0.0, 1 - abs(err) / actual)
        out.append({"id": i, "input_filename": r["input_filename"], "timestamp": r["timestamp"],
                    "predicted": pred, "actual": actual, "error": err,
                    "abs_error": abs(err), "accuracy": round(acc * 100, 1)})
    return out

# ---------- pages and basic endpoints ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return open(os.path.join(BASE, "index.html"), encoding="utf-8").read()

@app.get("/api/classes")
def classes():
    return ALL_CLASSES

# ---------- counting ----------
@app.post("/api/count")
def count(file: UploadFile = File(...),
          classes: str = Form(...),
          conf: float = Form(0.25),
          imgsz: int = Form(640)):
    name = file.filename or "image"
    if not name.lower().endswith(VALID_EXT):
        raise HTTPException(400, "Unsupported file type. Upload a JPG, PNG, BMP or WEBP image.")
    data = file.file.read()
    if len(data) == 0:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_MB} MB).")
    selected = [c for c in classes.split(",") if c in NAME_TO_ID]
    if not selected:
        raise HTTPException(400, "Select at least one valid class to count.")
    if not 0.05 <= conf <= 0.95:
        raise HTTPException(400, "Confidence must be between 0.05 and 0.95.")
    if imgsz not in (640, 960, 1280):
        raise HTTPException(400, "Image size must be 640, 960 or 1280.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read this image. The file may be corrupted.")

    try:
        # Run core YOLO detection with optimal NMS iou threshold
        with INFER_LOCK:
            result = model(img, conf=conf, imgsz=imgsz, iou=0.5,
                           classes=[NAME_TO_ID[c] for c in selected],
                           augment=USE_TTA, verbose=False)[0]
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {e}")

    # Retrieve existing bounding boxes and detected classes
    boxes = result.boxes
    detected_labels = [model.names[int(b.cls)] for b in boxes]
    counts = dict(Counter(detected_labels))

    # Plot initial annotations
    annotated_img = result.plot()

    # Fallback Optimization: If person class is selected, use Face Cascade for close-up portrait photos
    if FACE_FALLBACK and "person" in selected:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        added_faces = 0
        yolo_person_boxes = [b.xyxy[0].cpu().numpy() for b in boxes if model.names[int(b.cls)] == "person"]

        for (fx, fy, fw, fh) in faces:
            # Check if this face is already enclosed/contained in a YOLO person bounding box
            is_already_detected = False
            for px1, py1, px2, py2 in yolo_person_boxes:
                # Face center coordinates
                fcx, fcy = fx + fw // 2, fy + fh // 2
                if px1 <= fcx <= px2 and py1 <= fcy <= py2:
                    is_already_detected = True
                    break

            # If it's a new face detection (e.g. a cropped close-up or regular portrait photo YOLO missed)
            if not is_already_detected:
                added_faces += 1
                # Draw boundary around face as fallback person
                cv2.rectangle(annotated_img, (fx, fy), (fx + fw, fy + fh), (0, 255, 128), 3)
                cv2.putText(annotated_img, f"person (face-fallback) 0.85", (fx, max(fy - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)

        if added_faces > 0:
            counts["person"] = counts.get("person", 0) + added_faces

    total = sum(counts.values())
    ok, buf = cv2.imencode(".jpg", annotated_img)
    image_url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with LOG_LOCK:
        rows = read_rows()
        rows.append({"input_filename": name, "timestamp": ts, "confidence": conf,
                     "image_size": imgsz, "total_count": total,
                     "per_class_counts": "; ".join(f"{k}: {v}" for k, v in counts.items()),
                     "actual_count": ""})
        write_rows(rows)
        new_id = len(rows) - 1

    return {"id": new_id, "filename": name, "timestamp": ts, "conf": conf, "imgsz": imgsz,
            "total": total, "counts": counts, "image": image_url}

# ---------- history, ground truth, evaluation ----------
@app.get("/api/history")
def history():
    with LOG_LOCK:
        rows = read_rows()
    out = [dict(r, id=i) for i, r in enumerate(rows)]
    return out[-30:][::-1]

@app.post("/api/actual")
def set_actual(id: int = Form(...), actual: str = Form("")):
    actual = actual.strip()
    if actual != "" and not actual.isdigit():
        raise HTTPException(400, "Actual count must be a whole number (0 or more).")
    with LOG_LOCK:
        rows = read_rows()
        if not 0 <= id < len(rows):
            raise HTTPException(404, "Result not found.")
        rows[id]["actual_count"] = actual
        write_rows(rows)
    return {"ok": True}

@app.delete("/api/delete/{id}")
def delete_history(id: int):
    with LOG_LOCK:
        rows = read_rows()
        if not 0 <= id < len(rows):
            raise HTTPException(404, "Result not found.")
        # Delete by popping or filtering out
        rows.pop(id)
        write_rows(rows)
    return {"ok": True}

@app.get("/api/evaluation")
def evaluation():
    with LOG_LOCK:
        ev = evaluated(read_rows())
    n = len(ev)
    summary = {"n": n, "mae": 0, "mean_accuracy": 0, "exact": 0, "total_pred": 0, "total_actual": 0}
    if n:
        summary = {"n": n,
                   "mae": round(sum(r["abs_error"] for r in ev) / n, 2),
                   "mean_accuracy": round(sum(r["accuracy"] for r in ev) / n, 1),
                   "exact": sum(1 for r in ev if r["error"] == 0),
                   "total_pred": sum(r["predicted"] for r in ev),
                   "total_actual": sum(r["actual"] for r in ev)}
    return {"summary": summary, "rows": ev[::-1]}

# ---------- exports ----------
@app.get("/api/export")
def export():
    if not os.path.exists(LOG_PATH):
        raise HTTPException(404, "No results logged yet.")
    return FileResponse(LOG_PATH, media_type="text/csv", filename="results_log.csv")

@app.get("/api/export_eval")
def export_eval():
    with LOG_LOCK:
        ev = evaluated(read_rows())
    if not ev:
        raise HTTPException(404, "No evaluated images yet. Enter the actual counts first.")
    with open(EVAL_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input_filename", "timestamp", "predicted_count", "actual_count",
                    "error", "absolute_error", "accuracy_percent"])
        for r in ev[::-1]:
            w.writerow([r["input_filename"], r["timestamp"], r["predicted"], r["actual"],
                        r["error"], r["abs_error"], r["accuracy"]])
        n = len(ev)
        w.writerow([])
        w.writerow(["Images evaluated", n])
        w.writerow(["Mean absolute error (MAE)", round(sum(r["abs_error"] for r in ev) / n, 2)])
        w.writerow(["Mean accuracy (%)", round(sum(r["accuracy"] for r in ev) / n, 1)])
    return FileResponse(EVAL_PATH, media_type="text/csv", filename="evaluation_table.csv")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    if ON_CLOUD:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    else:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
