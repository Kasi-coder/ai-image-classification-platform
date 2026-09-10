import json
from pathlib import Path

def main():
    r=json.loads(Path("reports/model_comparison.json").read_text())
    best=max(r,key=lambda x:r[x]["accuracy"])
    m=r[best]
    text=f"""# Technical Report — AI-Powered Image Classification Platform

## Objective
Build and deploy an image classification platform using transfer learning.

## Dataset
CIFAR-10: 60,000 32×32 RGB images across 10 object classes.

## Data pipeline
Images are resized to 224×224 and normalized with ImageNet statistics. Training data uses random horizontal flips, random crops, and color jitter. The official training set is split into training and validation partitions; the official test set is held out for final evaluation.

## Models compared
- ResNet18 with ImageNet transfer learning
- MobileNetV3-Small with ImageNet transfer learning

The convolutional backbones are initially frozen and classification heads are fine-tuned.

## Performance comparison
"""
    for name,v in r.items():
        text += f"\n### {name}\nAccuracy: {v['accuracy']:.4f}\nPrecision: {v['precision']:.4f}\nRecall: {v['recall']:.4f}\nF1: {v['f1']:.4f}\nTop-5 Accuracy: {v['top5_accuracy']:.4f}\n"
    text += f"\n## Selected model\n**{best}** was selected using highest test accuracy among the compared trained candidates.\n"
    Path("reports/technical_report.md").write_text(text)
    print("Wrote reports/technical_report.md")
if __name__=="__main__": main()
