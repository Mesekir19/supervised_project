import os
import json
import math
import random
import logging
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


CONFIG = {
    "data_path":        "./nancy_data/nancy_poi_pairs.jsonl",
    "output_dir":       "./stage1_output",
    "min_confidence":   0.75,

    "train_ratio":      0.90,
    "val_ratio":        0.05,
    "test_ratio":       0.05,
    "seed":             42,

    "input_dim":        6,
    "hidden_dims":      [256, 128, 64],
    "dropout":          0.3,
    "num_classes":      4,

    "epochs":           50,
    "batch_size":       512,
    "learning_rate":    1e-3,
    "weight_decay":     1e-4,
    "patience":         7,
    "lr_patience":      3,
    "lr_factor":        0.5,
    "num_workers":      4,
}

LABELS = ["NORTH", "EAST", "SOUTH", "WEST"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for l, i in LABEL2IDX.items()}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

os.makedirs(CONFIG["output_dir"], exist_ok=True)
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])


def load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info(f"Loaded {len(records):,} records from {path}")
    return records


def build_features(record: dict) -> np.ndarray:
    la, lo_a = record["poi_a"]["lat"], record["poi_a"]["lon"]
    lb, lo_b = record["poi_b"]["lat"], record["poi_b"]["lon"]
    return np.array([la, lo_a, lb, lo_b, lb - la, lo_b - lo_a], dtype=np.float32)


def split_by_poi_pair(records: list, cfg: dict) -> tuple:
    log.info("Splitting by POI pair to avoid data leakage...")

    pair_groups = defaultdict(list)
    for r in records:
        if r["confidence_score"] < cfg["min_confidence"]:
            continue
        key = tuple(sorted([r["poi_a"]["name"], r["poi_b"]["name"]]))
        pair_groups[key].append(r)

    pairs = list(pair_groups.keys())
    random.shuffle(pairs)

    n = len(pairs)
    n_val  = max(1, int(n * cfg["val_ratio"]))
    n_test = max(1, int(n * cfg["test_ratio"]))
    n_train = n - n_val - n_test

    train_pairs = set(pairs[:n_train])
    val_pairs   = set(pairs[n_train:n_train + n_val])
    test_pairs  = set(pairs[n_train + n_val:])

    train, val, test = [], [], []
    for key, recs in pair_groups.items():
        if key in train_pairs:   train.extend(recs)
        elif key in val_pairs:   val.extend(recs)
        else:                    test.extend(recs)

    log.info(f"  Unique pairs   — train: {len(train_pairs):,}  val: {len(val_pairs):,}  test: {len(test_pairs):,}")
    log.info(f"  Records        — train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        counts = Counter(r["cardinal_direction"] for r in split_data)
        log.info(f"  {split_name} distribution: {dict(counts)}")

    return train, val, test


def compute_scaler(records: list) -> tuple:
    X = np.stack([build_features(r) for r in records])
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def save_scaler(mean, std, path):
    obj = {"mean": mean.tolist(), "std": std.tolist()}
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    log.info(f"Scaler saved → {path}")


class POIPairDataset(Dataset):
    def __init__(self, records: list, mean: np.ndarray, std: np.ndarray):
        self.X = []
        self.y = []
        for r in records:
            feat = (build_features(r) - mean) / std
            self.X.append(feat)
            self.y.append(LABEL2IDX[r["cardinal_direction"]])
        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_weighted_sampler(dataset: POIPairDataset) -> WeightedRandomSampler:
    counts = Counter(dataset.y.tolist())
    total = len(dataset)
    weights = [total / counts[int(label)] for label in dataset.y]
    return WeightedRandomSampler(weights, num_samples=total, replacement=True)


class CardinalMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, num_classes: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, optimizer, criterion, device) -> tuple:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += len(y)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def plot_training_curves(history: dict, path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train", linewidth=2)
    ax1.plot(history["val_loss"],   label="Val",   linewidth=2, linestyle="--")
    ax1.set_title("Loss", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot([a * 100 for a in history["train_acc"]], label="Train", linewidth=2)
    ax2.plot([a * 100 for a in history["val_acc"]],   label="Val",   linewidth=2, linestyle="--")
    ax2.set_title("Accuracy", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Stage 1 — Cardinal Direction Classifier", fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Training curves saved → {path}")


def plot_confusion_matrix(labels, preds, path: str):
    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (% per true class)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrix saved → {path}")


def load_model_for_inference(output_dir: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(output_dir, "scaler_params.json")) as f:
        scaler = json.load(f)
    mean = np.array(scaler["mean"], dtype=np.float32)
    std  = np.array(scaler["std"],  dtype=np.float32)

    model = CardinalMLP(
        input_dim=CONFIG["input_dim"],
        hidden_dims=CONFIG["hidden_dims"],
        num_classes=CONFIG["num_classes"],
        dropout=0.0
    )
    ckpt = torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    def predict(lat_a, lon_a, lat_b, lon_b):
        raw = np.array([lat_a, lon_a, lat_b, lon_b, lat_b - lat_a, lon_b - lon_a], dtype=np.float32)
        x = torch.tensor((raw - mean) / std).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1).squeeze().cpu().tolist()
        idx = int(np.argmax(probs))
        return {
            "direction":     IDX2LABEL[idx],
            "confidence":    round(probs[idx], 4),
            "probabilities": {IDX2LABEL[i]: round(p, 4) for i, p in enumerate(probs)}
        }

    return predict


def main():
    cfg = CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    records = load_jsonl(cfg["data_path"])
    train_recs, val_recs, test_recs = split_by_poi_pair(records, cfg)

    mean, std = compute_scaler(train_recs)
    save_scaler(mean, std, os.path.join(cfg["output_dir"], "scaler_params.json"))

    train_ds = POIPairDataset(train_recs, mean, std)
    val_ds   = POIPairDataset(val_recs,   mean, std)
    test_ds  = POIPairDataset(test_recs,  mean, std)

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, num_workers=cfg["num_workers"])
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"])
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=cfg["num_workers"])

    model = CardinalMLP(
        input_dim=cfg["input_dim"],
        hidden_dims=cfg["hidden_dims"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=cfg["lr_patience"],
        factor=cfg["lr_factor"]
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(cfg["output_dir"], "best_model.pt")

    log.info("Starting training...")
    log.info(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Loss':>10}  {'Val Acc':>10}  {'LR':>10}")
    log.info("─" * 65)

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        log.info(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc*100:>9.2f}%  "
                 f"{val_loss:>10.4f}  {val_acc*100:>9.2f}%  {lr_now:>10.2e}")
        if lr_now < prev_lr:
            log.info(f"  LR reduced: {prev_lr:.2e} -> {lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            log.info(f"  ✓ New best val accuracy: {best_val_acc*100:.2f}% — model saved")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                log.info(f"Early stopping at epoch {epoch} (no improvement for {cfg['patience']} epochs)")
                break

    torch.save(model.state_dict(), os.path.join(cfg["output_dir"], "final_model.pt"))

    log.info("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)

    log.info(f"\n{'='*45}")
    log.info(f"  TEST RESULTS")
    log.info(f"{'='*45}")
    log.info(f"  Loss:     {test_loss:.4f}")
    log.info(f"  Accuracy: {test_acc*100:.2f}%")
    log.info(f"\n{classification_report(test_labels, test_preds, target_names=LABELS)}")

    report = {
        "config":        cfg,
        "best_val_acc":  round(best_val_acc, 6),
        "test_loss":     round(test_loss, 6),
        "test_accuracy": round(test_acc, 6),
        "epochs_trained": len(history["train_loss"]),
        "history":       history,
    }
    report_path = os.path.join(cfg["output_dir"], "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Training report saved → {report_path}")

    with open(os.path.join(cfg["output_dir"], "label_encoder.json"), "w") as f:
        json.dump({"label2idx": LABEL2IDX, "idx2label": IDX2LABEL}, f, indent=2)

    plot_training_curves(history, os.path.join(cfg["output_dir"], "training_curves.png"))
    plot_confusion_matrix(test_labels, test_preds, os.path.join(cfg["output_dir"], "confusion_matrix.png"))

    log.info("\nInference demo (Brasserie Excelsior vs Place Commanderie):")
    predict = load_model_for_inference(cfg["output_dir"], device)
    result = predict(lat_a=48.6921, lon_a=6.1844, lat_b=48.6912, lon_b=6.1801)
    log.info(f"  {json.dumps(result, indent=2)}")

    log.info(f"\n All outputs saved to: {cfg['output_dir']}/")


if __name__ == "__main__":
    main()
