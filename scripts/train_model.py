"""
train_model.py
--------------
Trains a 1D CNN (or ConvLSTM) on the breathing dataset using
Leave-One-Participant-Out (LOPO) cross-validation.

Usage:
    python scripts/train_model.py -dataset Dataset/breathing_dataset.csv
    python scripts/train_model.py -dataset Dataset/breathing_dataset.csv --model conv_lstm
    python scripts/train_model.py -dataset Dataset/breathing_dataset.csv --epochs 30 --lr 1e-3
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              classification_report, confusion_matrix)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── make sure models/ is importable ───────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.cnn_model      import build_cnn
from models.conv_lstm_model import build_conv_lstm


# ── dataset construction ──────────────────────────────────────────────────────

FEATURE_COLS_PREFIX = [
    "nasal_", "thoracic_", "spo2_", "nasal_thoracic"
]

def get_feature_cols(df):
    return [c for c in df.columns
            if any(c.startswith(p) for p in FEATURE_COLS_PREFIX)]


def make_tensors(X, y, device):
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.long).to(device)
    return Xt, yt


# ── training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss   = criterion(logits, yb)
        total_loss += loss.item() * len(yb)
        preds  = logits.argmax(1)
        correct += (preds == yb).sum().item()
        n += len(yb)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    return total_loss / n, correct / n, np.array(all_preds), np.array(all_labels)


# ── confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(cm, classes, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset",  default="Dataset/breathing_dataset.csv")
    parser.add_argument("--model",   default="cnn",
                        choices=["cnn", "conv_lstm"], help="Model architecture")
    parser.add_argument("--epochs",  type=int, default=25)
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--lr",      type=float, default=5e-4)
    parser.add_argument("--out_dir", default="Results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.dataset):
        sys.exit(f"[ERROR] Dataset not found: {args.dataset}\n"
                 "Run create_dataset.py first.")

    df = pd.read_csv(args.dataset)
    feat_cols = get_feature_cols(df)
    print(f"Loaded {len(df)} windows, {len(feat_cols)} features, "
          f"labels: {df['label'].value_counts().to_dict()}")

    # ── encode labels ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    classes = le.classes_
    n_classes = len(classes)
    print(f"Classes: {list(classes)}")

    participants = sorted(df["participant"].unique())
    print(f"Participants: {participants}\n")

    # ── class weights for imbalanced data ────────────────────────────────────
    counts = df["label_enc"].value_counts().sort_index().values
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # ── LOPO cross-validation ─────────────────────────────────────────────────
    fold_results = []
    all_preds_global  = []
    all_labels_global = []

    # ── feature dimensionality for CNN signal input ───────────────────────────
    # We treat feature vector as a 1D sequence (1 channel, N features)
    # This is a feature-based CNN – simple and effective for tabular features.
    n_feats   = len(feat_cols)
    seq_len   = n_feats
    n_channels = 1

    for fold_idx, test_pid in enumerate(participants):
        print(f"── Fold {fold_idx+1}/{len(participants)}: Test = {test_pid} ──")

        train_df = df[df["participant"] != test_pid].copy()
        test_df  = df[df["participant"] == test_pid].copy()

        # scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feat_cols].values).astype(np.float32)
        X_test  = scaler.transform(test_df[feat_cols].values).astype(np.float32)
        y_train = train_df["label_enc"].values
        y_test  = test_df["label_enc"].values

        # reshape to (N, 1, features) for 1D CNN
        X_train = X_train[:, np.newaxis, :]
        X_test  = X_test[:, np.newaxis, :]

        train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  drop_last=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

        # ── build model ───────────────────────────────────────────────────────
        if args.model == "cnn":
            model = build_cnn(n_channels=n_channels,
                              seq_len=seq_len,
                              n_classes=n_classes,
                              dropout=0.3).to(device)
        else:
            model = build_conv_lstm(n_channels=n_channels,
                                    seq_len=seq_len,
                                    n_classes=n_classes,
                                    dropout=0.3).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_val_loss = float("inf")
        best_state    = None
        patience      = 8
        no_improve    = 0

        # ── training ──────────────────────────────────────────────────────────
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            va_loss, va_acc, _, _ = eval_epoch(model, test_loader, criterion, device)
            scheduler.step()

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{args.epochs}  "
                      f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
                      f"va_loss={va_loss:.4f} va_acc={va_acc:.3f}  "
                      f"({time.time()-t0:.1f}s)")

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # ── evaluate best model ───────────────────────────────────────────────
        model.load_state_dict(best_state)
        _, _, preds, labels = eval_epoch(model, test_loader, criterion, device)

        acc  = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average="macro", zero_division=0)
        rec  = recall_score(labels, preds,    average="macro", zero_division=0)
        f1   = f1_score(labels, preds,        average="macro", zero_division=0)
        cm   = confusion_matrix(labels, preds, labels=list(range(n_classes)))

        print(f"\n  [Fold {fold_idx+1}] Acc={acc:.4f}  Prec={prec:.4f}  "
              f"Rec={rec:.4f}  F1={f1:.4f}")
        print(classification_report(labels, preds, target_names=classes, zero_division=0))

        fold_results.append({
            "fold": fold_idx + 1,
            "test_participant": test_pid,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        })
        all_preds_global.extend(preds.tolist())
        all_labels_global.extend(labels.tolist())

        # per-fold confusion matrix
        cm_path = os.path.join(args.out_dir, f"cm_fold{fold_idx+1}_{test_pid}.png")
        plot_confusion_matrix(cm, classes,
                              f"Fold {fold_idx+1} – Test: {test_pid}", cm_path)

        # save model checkpoint
        ckpt_path = os.path.join(args.out_dir, f"model_fold{fold_idx+1}_{test_pid}.pt")
        torch.save({"model_state": best_state,
                    "label_encoder": list(classes),
                    "fold": fold_idx + 1,
                    "test_pid": test_pid}, ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path}\n")

    # ── aggregate results ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("LOPO Cross-Validation Summary")
    print("═" * 60)
    results_df = pd.DataFrame(fold_results)
    print(results_df.to_string(index=False))

    print(f"\nMean  Accuracy  = {results_df['accuracy'].mean():.4f} "
          f"± {results_df['accuracy'].std():.4f}")
    print(f"Mean  Precision = {results_df['precision'].mean():.4f} "
          f"± {results_df['precision'].std():.4f}")
    print(f"Mean  Recall    = {results_df['recall'].mean():.4f} "
          f"± {results_df['recall'].std():.4f}")
    print(f"Mean  F1        = {results_df['f1'].mean():.4f} "
          f"± {results_df['f1'].std():.4f}")

    results_df.to_csv(os.path.join(args.out_dir, "lopo_results.csv"), index=False)

    # ── overall confusion matrix ───────────────────────────────────────────────
    global_cm  = confusion_matrix(all_labels_global, all_preds_global,
                                  labels=list(range(n_classes)))
    overall_cm_path = os.path.join(args.out_dir, "confusion_matrix_overall.png")
    plot_confusion_matrix(global_cm, classes,
                          "Overall LOPO Confusion Matrix", overall_cm_path)
    print(f"\nOverall confusion matrix → {overall_cm_path}")

    # ── bar chart of per-fold metrics ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    width = 0.2
    for i, (metric, color) in enumerate(zip(
            ["accuracy", "precision", "recall", "f1"],
            ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"])):
        ax.bar(x + i * width, results_df[metric], width,
               label=metric.capitalize(), color=color, alpha=0.85)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"Fold {i+1}\n({p})"
                        for i, p in enumerate(results_df["test_participant"])],
                       fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"LOPO CV – Per-Fold Metrics ({args.model.upper()})")
    ax.legend()
    ax.axhline(results_df["accuracy"].mean(), color="#3498db",
               linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    metrics_path = os.path.join(args.out_dir, "per_fold_metrics.png")
    plt.savefig(metrics_path, dpi=150)
    plt.close(fig)
    print(f"Per-fold metrics chart → {metrics_path}")
    print(f"\nAll results saved in '{args.out_dir}/'")


if __name__ == "__main__":
    main()
