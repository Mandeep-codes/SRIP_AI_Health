"""
train_sklearn.py  (optional fallback)
--------------------------------------
Trains an sklearn Random Forest / Gradient Boosting using LOPO CV.
Useful for quick evaluation if PyTorch is not yet installed,
or as a baseline to compare against the CNN.

Usage:
    python scripts/train_sklearn.py -dataset Dataset/breathing_dataset.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight

FEATURE_PREFIXES = ["nasal_", "thoracic_", "spo2_", "nasal_thoracic"]


def get_feature_cols(df):
    return [c for c in df.columns
            if any(c.startswith(p) for p in FEATURE_PREFIXES)]


def plot_cm(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", default="Dataset/breathing_dataset.csv")
    parser.add_argument("--model",  default="rf",
                        choices=["rf", "gb"], help="rf=RandomForest, gb=GradientBoosting")
    parser.add_argument("--out_dir", default="Results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.dataset):
        sys.exit(f"[ERROR] Dataset not found: {args.dataset}")

    df = pd.read_csv(args.dataset)
    feat_cols  = get_feature_cols(df)
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    classes    = le.classes_
    n_classes  = len(classes)
    participants = sorted(df["participant"].unique())

    print(f"Dataset: {len(df)} windows | features: {len(feat_cols)} | "
          f"classes: {list(classes)}")
    print(f"Participants: {participants}\n")

    fold_results = []
    all_preds, all_labels = [], []

    for fold_idx, test_pid in enumerate(participants):
        print(f"── Fold {fold_idx+1}/{len(participants)}: Test = {test_pid} ──")
        train_df = df[df["participant"] != test_pid]
        test_df  = df[df["participant"] == test_pid]

        sc = StandardScaler()
        X_train = sc.fit_transform(train_df[feat_cols].fillna(0).values)
        X_test  = sc.transform(test_df[feat_cols].fillna(0).values)
        y_train = train_df["label_enc"].values
        y_test  = test_df["label_enc"].values

        sw = compute_sample_weight("balanced", y_train)

        if args.model == "rf":
            clf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                         class_weight="balanced",
                                         n_jobs=-1, random_state=42)
        else:
            clf = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                              learning_rate=0.05, random_state=42)

        clf.fit(X_train, y_train, sample_weight=sw if args.model == "gb" else None
                if args.model == "rf" else sw)
        preds = clf.predict(X_test)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec  = recall_score(y_test,  preds, average="macro", zero_division=0)
        f1   = f1_score(y_test,     preds, average="macro", zero_division=0)
        cm   = confusion_matrix(y_test, preds, labels=list(range(n_classes)))

        print(f"  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
        print(classification_report(y_test, preds, target_names=classes, zero_division=0))

        fold_results.append({"fold": fold_idx+1, "test_participant": test_pid,
                              "accuracy": acc, "precision": prec,
                              "recall": rec, "f1": f1})
        all_preds.extend(preds.tolist()); all_labels.extend(y_test.tolist())

        plot_cm(cm, classes,
                f"Fold {fold_idx+1} – {test_pid} ({args.model.upper()})",
                os.path.join(args.out_dir, f"sklearn_cm_fold{fold_idx+1}.png"))

    results_df = pd.DataFrame(fold_results)
    print("\n" + "═"*55)
    print("LOPO Summary")
    print("═"*55)
    print(results_df.to_string(index=False))
    for m in ["accuracy","precision","recall","f1"]:
        print(f"  Mean {m:10s} = {results_df[m].mean():.4f} ± {results_df[m].std():.4f}")

    results_df.to_csv(os.path.join(args.out_dir, f"lopo_results_{args.model}.csv"), index=False)

    global_cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
    plot_cm(global_cm, classes,
            f"Overall LOPO ({args.model.upper()})",
            os.path.join(args.out_dir, f"sklearn_cm_overall_{args.model}.png"))
    print(f"\nResults saved in '{args.out_dir}/'")


if __name__ == "__main__":
    main()
