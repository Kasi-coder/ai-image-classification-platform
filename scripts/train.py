import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from src.data import make_loaders, CLASSES
from src.models import build_model, freeze_backbone
from src.metrics import classification_metrics

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        if train: optimizer.zero_grad()
        out=model(x); loss=criterion(out,y)
        if train:
            loss.backward(); optimizer.step()
        loss_sum += loss.item()*x.size(0)
        correct += (out.argmax(1)==y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def probabilities(model, loader, device):
    model.eval(); ys=[]; ps=[]
    for x,y in loader:
        out=torch.softmax(model(x.to(device)),1).cpu().numpy()
        ps.append(out); ys.extend(y.numpy())
    return np.array(ys), np.vstack(ps)

def train_one(name, train_loader, val_loader, test_loader, epochs, device):
    model=build_model(name, pretrained=True)
    model=freeze_backbone(model,name).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
    best=0
    for ep in range(1,epochs+1):
        trl,tra=run_epoch(model,train_loader,criterion,optimizer,device,True)
        vall,vala=run_epoch(model,val_loader,criterion,optimizer,device,False)
        history["train_loss"].append(trl); history["train_acc"].append(tra)
        history["val_loss"].append(vall); history["val_acc"].append(vala)
        print(f"{name} epoch {ep}/{epochs}: train_acc={tra:.4f} val_acc={vala:.4f}")
        if vala>best:
            best=vala
            best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    y,p=probabilities(model,test_loader,device)
    metrics=classification_metrics(y,p)
    return model, metrics, history

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=3)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--train-limit",type=int,default=10000)
    ap.add_argument("--val-limit",type=int,default=2000)
    ap.add_argument("--test-limit",type=int,default=2000)
    args=ap.parse_args()
    Path("models").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    print("Device:",device)
    loaders=make_loaders(batch_size=args.batch_size, train_limit=args.train_limit, val_limit=args.val_limit, test_limit=args.test_limit)
    train_loader,val_loader,test_loader=loaders
    results={}; histories={}
    best_name=None; best_acc=-1; best_model=None; best_metrics=None
    for name in ["resnet18","mobilenet_v3_small"]:
        t=time.time()
        model,metrics,hist=train_one(name,*loaders,args.epochs,device)
        results[name]=metrics; histories[name]=hist
        results[name]["training_seconds"]=time.time()-t
        if metrics["accuracy"]>best_acc:
            best_acc=metrics["accuracy"]; best_name=name; best_model=model; best_metrics=metrics
    torch.save({"model_state_dict":best_model.state_dict(),"model_name":best_name,"classes":CLASSES},"models/best_model.pt")
    Path("models/model_metadata.json").write_text(json.dumps({"model_name":best_name,"classes":CLASSES,"device":device,"epochs":args.epochs},indent=2))
    Path("reports/model_comparison.json").write_text(json.dumps(results,indent=2))
    Path("reports/best_model_metrics.json").write_text(json.dumps(best_metrics,indent=2))
    Path("reports/training_history.json").write_text(json.dumps(histories,indent=2))
    print(f"BEST MODEL: {best_name} | test accuracy={best_acc:.4f}")

if __name__=="__main__":
    main()
