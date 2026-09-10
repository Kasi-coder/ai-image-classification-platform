from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from src.inference import load_bundle, predict_image
from src.database import init_db, save_prediction, recent_predictions

app = FastAPI(title="AI Image Classification API", version="1.0.0")
_model = None
_tf = None

@app.on_event("startup")
def startup():
    global _model, _tf
    init_db()
    try:
        _model, _tf = load_bundle()
    except FileNotFoundError:
        _model, _tf = None, None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.get("/classes")
def classes():
    from src.data import CLASSES
    return {"classes": CLASSES}

@app.get("/history")
def history(limit: int = 20):
    return {"items": [
        {"filename": r[0], "predicted_class": r[1], "confidence": r[2],
         "top_predictions": r[3], "created_at": r[4]}
        for r in recent_predictions(limit)
    ]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(503, "Trained model not found. Run scripts/train.py first.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")
    data = await file.read()
    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}")
    results = predict_image(image, _model, _tf, top_k=5)
    save_prediction(file.filename or "upload", results[0]["class"], results[0]["confidence"], str(results))
    return {"filename": file.filename, "prediction": results[0], "top_5": results}
