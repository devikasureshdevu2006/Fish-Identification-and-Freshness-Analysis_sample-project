import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def collect_samples(data_dirs: list[str], task: str):
    dir_paths = [Path(d) for d in data_dirs]
    for p in dir_paths:
        if not p.exists():
            raise FileNotFoundError(f"Dataset directory not found: {p}")

    samples: list[tuple[str, str]] = []

    if task == "freshness":
        for p in dir_paths:
            for img_file in p.rglob("*"):
                if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                    parent_name = img_file.parent.name.lower()
                    if parent_name in {
                        "non-fresh",
                        "non_fresh",
                        "not fresh",
                        "not_fresh",
                        "spoiled",
                        "stale",
                    }:
                        samples.append((str(img_file), "Non-Fresh"))
                    elif parent_name == "fresh":
                        samples.append((str(img_file), "Fresh"))
    else:
        for p in dir_paths:
            for sub in sorted(p.iterdir()):
                if sub.is_dir():
                    for img_file in sub.rglob("*"):
                        if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                            samples.append((str(img_file), sub.name))

    classes = sorted({cls_name for _, cls_name in samples})
    if len(classes) < 2:
        raise ValueError(f"Need at least 2 classes for task '{task}', found: {classes}")
    if len(samples) < 3:
        raise ValueError(
            f"Need at least 3 images for 70/15/15 split, found {len(samples)}."
        )

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    indexed_samples = [
        (img_path, class_to_idx[cls_name]) for img_path, cls_name in samples
    ]
    return classes, indexed_samples


class FishDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
        return rgb_img, label


def split_70_15_15(total_len: int, seed: int = 42):
    n_train = max(1, int(0.70 * total_len))
    n_val = max(1, int(0.15 * total_len))
    n_test = max(1, total_len - n_train - n_val)

    while n_train + n_val + n_test > total_len:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1

    perm = torch.randperm(
        total_len, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    return (
        perm[:n_train],
        perm[n_train : n_train + n_val],
        perm[n_train + n_val : n_train + n_val + n_test],
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = torch.argmax(model(images), dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="freshness", choices=["species", "freshness"]
    )
    parser.add_argument("--data_dir", type=str, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="models")
    args = parser.parse_args()

    classes, samples = collect_samples(args.data_dir, args.task)

    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_idx, val_idx, test_idx = split_70_15_15(len(samples), seed=args.seed)
    train_ds = Subset(FishDataset(samples, train_tfm), train_idx)
    val_ds = Subset(FishDataset(samples, eval_tfm), val_idx)
    test_ds = Subset(FishDataset(samples, eval_tfm), test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        val_true, val_pred = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_true, val_pred) * 100.0
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch [{epoch}/{args.epochs}] - Loss: {running_loss / max(1, len(train_ds)):.4f} - Val Acc: {val_acc:.2f}%"
        )

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = {
        "task": args.task,
        "classes": classes,
        "split_counts": {
            "total": len(samples),
            "train_70": len(train_ds),
            "val_15": len(val_ds),
            "test_15": len(test_ds),
        },
        "accuracy": round(float(accuracy_score(y_true, y_pred) * 100.0), 2),
        "precision": round(
            float(
                precision_score(
                    y_true, y_pred, average="weighted", zero_division=0
                )
                * 100.0
            ),
            2,
        ),
        "recall": round(
            float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
                * 100.0
            ),
            2,
        ),
        "f1_score": round(
            float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
                * 100.0
            ),
            2,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(classes)))
        ).tolist(),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.task}_model.pth"
    metrics_path = out_dir / f"{args.task}_metrics.json"

    torch.save(
        {"task": args.task, "classes": classes, "state_dict": model.state_dict()},
        model_path,
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()