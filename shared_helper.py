"""Shared helper utilities for Part 2 and Part 4."""
import os
import csv
import numpy as np


def load_and_split():
    data_path = os.path.join("data", "records.csv")
    with open(data_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    class0 = [r for r in rows if int(r[8]) == 0]
    class1 = [r for r in rows if int(r[8]) == 1]

    n0_train = (7 * len(class0)) // 10  # 441
    n1_train = (7 * len(class1)) // 10  # 119

    train_raw = class0[:n0_train] + class1[:n1_train]
    test_raw = class0[n0_train:] + class1[n1_train:]

    # Strictly restore original file order
    file_order = {r[0]: i for i, r in enumerate(rows)}
    train_raw.sort(key=lambda r: file_order[r[0]])
    test_raw.sort(key=lambda r: file_order[r[0]])

    X_train_orig = np.array([[float(r[j]) for j in range(1, 8)] for r in train_raw], dtype=np.float64)
    y_train = np.array([int(r[8]) for r in train_raw], dtype=np.int32)

    X_test_orig = np.array([[float(r[j]) for j in range(1, 8)] for r in test_raw], dtype=np.float64)
    y_test = np.array([int(r[8]) for r in test_raw], dtype=np.int32)

    mean_tr = np.mean(X_train_orig, axis=0)
    std_tr = np.std(X_train_orig, axis=0, ddof=1)

    X_train_std = (X_train_orig - mean_tr) / std_tr
    X_test_std = (X_test_orig - mean_tr) / std_tr

    return {
        "train_raw": train_raw, "test_raw": test_raw,
        "X_train_orig": X_train_orig, "X_test_orig": X_test_orig,
        "X_train_std": X_train_std, "X_test_std": X_test_std,
        "y_train": y_train, "y_test": y_test,
        "mean_tr": mean_tr, "std_tr": std_tr
    }


def confusion_metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec,
        "specificity": spec, "fpr": fpr, "f1": f1
    }