from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def build_transforms(image_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def make_loaders(root="data/raw", batch_size=64, val_fraction=0.1,
                 train_limit=None, val_limit=None, test_limit=None, seed=42):
    train_tf, eval_tf = build_transforms()
    base = datasets.CIFAR10(root=root, train=True, download=True, transform=None)
    test_base = datasets.CIFAR10(root=root, train=False, download=True, transform=None)

    g = torch.Generator().manual_seed(seed)
    n_val = int(len(base) * val_fraction)
    n_train = len(base) - n_val
    train_idx, val_idx = random_split(range(len(base)), [n_train, n_val], generator=g)
    train_idx, val_idx = list(train_idx), list(val_idx)

    if train_limit:
        train_idx = train_idx[:min(train_limit, len(train_idx))]
    if val_limit:
        val_idx = val_idx[:min(val_limit, len(val_idx))]
    test_idx = list(range(len(test_base)))
    if test_limit:
        test_idx = test_idx[:min(test_limit, len(test_idx))]

    train_ds = Subset(datasets.CIFAR10(root=root, train=True, download=False, transform=train_tf), train_idx)
    val_ds = Subset(datasets.CIFAR10(root=root, train=True, download=False, transform=eval_tf), val_idx)
    test_ds = Subset(datasets.CIFAR10(root=root, train=False, download=False, transform=eval_tf), test_idx)

    kwargs = dict(batch_size=batch_size, num_workers=0, pin_memory=torch.cuda.is_available())
    return DataLoader(train_ds, shuffle=True, **kwargs), DataLoader(val_ds, shuffle=False, **kwargs), DataLoader(test_ds, shuffle=False, **kwargs)
