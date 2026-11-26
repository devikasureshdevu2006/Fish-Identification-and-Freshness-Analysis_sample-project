import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def build_cnn_model(num_classes: int) -> nn.Module:
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def split_70_15_15(dataset_dir: str, seed: int = 42):
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    base_ds = datasets.ImageFolder(dataset_dir)
    indices = np.arange(len(base_ds))
    targets = np.array(base_ds.targets)

    train_idx, temp_idx, _, temp_targets = train_test_split(
        indices, targets, test_size=0.30, stratify=targets, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_targets, random_state=seed
    )

    train_ds = Subset(datasets.ImageFolder(dataset_dir, transform=train_tfms), train_idx)
    val_ds = Subset(datasets.ImageFolder(dataset_dir, transform=eval_tfms), val_idx)
    test_ds = Subset(datasets.ImageFolder(dataset_dir, transform=eval_tfms), test_idx)
    return train_ds, val_ds, test_ds, base_ds.classes


def train_model(task: str, data_dir: str, out_dir: str = "models", epochs: int = 8, batch_size: int = 32, lr: float = 1e-3):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, test_ds, classes = split_70_15_15(data_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_cnn_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    ckpt_path = os.path.join(out_dir, f"{task}_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                preds = torch.argmax(model(imgs.to(device)), dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.numpy())

        val_acc = float(accuracy_score(val_true, val_preds))
        print(f"Epoch {epoch}/{epochs} | Loss: {running_loss / len(train_ds):.4f} | Val Acc: {val_acc:.4f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = torch.argmax(model(imgs.to(device)), dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "task": task,
        "classes": classes,
        "split": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "accuracy": round(acc * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "confusion_matrix": cm,
    }

    with open(os.path.join(out_dir, f"{task}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([{
        "Task": task,
        "Accuracy_%": metrics["accuracy"],
        "Precision_%": metrics["precision"],
        "Recall_%": metrics["recall"],
        "F1_Score_%": metrics["f1_score"],
    }]).to_csv(os.path.join(out_dir, f"{task}_metrics.csv"), index=False)

    print(f"Saved {ckpt_path} | Test Accuracy: {metrics['accuracy']}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["species", "freshness"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="models")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    train_model(args.task, args.data_dir, args.out_dir, args.epochs)