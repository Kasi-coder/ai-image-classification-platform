import argparse
import json
from pathlib import Path

import torch
from torch import nn, optim

from src.data import make_loaders
from src.models import build_model, freeze_backbone


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total, correct = 0, 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser(description="Small validation-based learning-rate tuning experiment")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--train-limit", type=int, default=5000)
    ap.add_argument("--val-limit", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rates = [1e-3, 3e-4]
    results = []

    print("Device:", device)
    print("Tuning MobileNetV3-Small learning rate on validation accuracy")

    for lr in learning_rates:
        train_loader, val_loader, _ = make_loaders(
            batch_size=args.batch_size,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
            test_limit=500,
        )
        model = build_model("mobilenet_v3_small", pretrained=True)
        model = freeze_backbone(model, "mobilenet_v3_small").to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )

        best_val = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
            best_val = max(best_val, val_acc)
            print(
                f"lr={lr:g} epoch {epoch}/{args.epochs}: "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )

        results.append({
            "learning_rate": lr,
            "weight_decay": 1e-4,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "best_validation_accuracy": best_val,
        })

    best = max(results, key=lambda x: x["best_validation_accuracy"])
    Path("reports").mkdir(exist_ok=True)
    Path("reports/hyperparameter_tuning.json").write_text(json.dumps({
        "model": "mobilenet_v3_small",
        "selection_metric": "validation_accuracy",
        "candidates": results,
        "best_configuration": best,
    }, indent=2))

    print("BEST CONFIGURATION:", json.dumps(best))
    print("Saved: reports/hyperparameter_tuning.json")


if __name__ == "__main__":
    main()
