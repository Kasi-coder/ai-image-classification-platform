import json
from pathlib import Path
import numpy as np
import torch
from src.data import make_loaders
from src.inference import DEVICE
from src.models import build_model
from src.metrics import classification_metrics

def main():
    meta=json.loads(Path("models/model_metadata.json").read_text())
    _,_,test=make_loaders(test_limit=None)
    model=build_model(meta["model_name"],pretrained=False)
    ck=torch.load("models/best_model.pt",map_location=DEVICE,weights_only=True)
    model.load_state_dict(ck["model_state_dict"]); model.to(DEVICE).eval()
    ys=[]; ps=[]
    with torch.no_grad():
        for x,y in test:
            p=torch.softmax(model(x.to(DEVICE)),1).cpu().numpy()
            ps.append(p); ys.extend(y.numpy())
    metrics=classification_metrics(np.array(ys),np.vstack(ps))
    Path("reports/best_model_metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
