import json
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from src.models import build_model
from src.data import CLASSES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_bundle(model_path="models/best_model.pt", metadata_path="models/model_metadata.json"):
    meta = json.loads(Path(metadata_path).read_text())
    model = build_model(meta["model_name"], pretrained=False)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return model, tf

def predict_image(image: Image.Image, model, tf, top_k=5):
    x = tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    vals, idx = torch.topk(probs, k=top_k)
    results = [{"class": CLASSES[int(i)], "confidence": float(v)} for v,i in zip(vals.cpu(), idx.cpu())]
    return results
