"""Part 2: Classification and evaluation."""
import os
import math
import numpy as np
from shared_helper import load_and_split, confusion_metrics

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def run():
    sp = load_and_split()
    X_tr = sp["X_train_std"]
    y_tr = sp["y_train"]
    X_te = sp["X_test_std"]
    y_te = sp["y_test"]
    n_tr = len(y_tr)
    n_te = len(y_te)

    # 2.2 Logistic regression
    w = np.zeros(7, dtype=np.float64)
    b = 0.0
    eta = 0.1
    losses = {}
    for epoch in range(1001):
        p = sigmoid(X_tr @ w + b)
        loss = - np.mean(y_tr * np.log(p + 1e-15) + (1 - y_tr) * np.log(1 - p + 1e-15))
        if epoch in [0, 1, 10, 100, 1000]:
            losses[epoch] = loss
        if epoch < 1000:
            grad_w = (1.0 / n_tr) * (X_tr.T @ (p - y_tr))
            grad_b = (1.0 / n_tr) * np.sum(p - y_tr)
            w -= eta * grad_w
            b -= eta * grad_b
    p_te = sigmoid(X_te @ w + b)
    test_bce = - np.mean(y_te * np.log(p_te + 1e-15) + (1 - y_te) * np.log(1 - p_te + 1e-15))

    # 2.3 Learning rates
    lr_losses = {}
    for lr in [0.01, 0.1, 1.0]:
        w_lr, b_lr = np.zeros(7), 0.0
        for _ in range(200):
            p_lr = sigmoid(X_tr @ w_lr + b_lr)
            grad_w = (1.0 / n_tr) * (X_tr.T @ (p_lr - y_tr))
            grad_b = (1.0 / n_tr) * np.sum(p_lr - y_tr)
            w_lr -= lr * grad_w
            b_lr -= lr * grad_b
        p_lr = sigmoid(X_tr @ w_lr + b_lr)
        lr_losses[lr] = - np.mean(y_tr * np.log(p_lr + 1e-15) + (1 - y_tr) * np.log(1 - p_lr + 1e-15))

    odds_ratios = np.exp(w)

    # 2.5 Evaluations
    m_05 = confusion_metrics(y_te, (p_te >= 0.5).astype(int))
    m_03 = confusion_metrics(y_te, (p_te >= 0.3).astype(int))
    blind_acc = float(np.mean(y_te == 0))

    # 2.6 Threshold sweep CSV
    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "part2_thresholds.csv"), "w") as f:
        f.write("tau,tp,fp,tn,fn,tpr,fpr\n")
        for k in range(1, 10):
            tau = round(k * 0.1, 1)
            cm = confusion_metrics(y_te, (p_te >= tau).astype(int))
            f.write(f"{tau:.1f},{cm['tp']},{cm['fp']},{cm['tn']},{cm['fn']},{cm['recall']:.4f},{cm['fpr']:.4f}\n")

        # 2.7 5-fold CV for KNN
        k_vals = list(range(1, 23, 2))
        cv_f1s = []
        fold_ids = np.array([i % 5 for i in range(n_tr)])

        for k in k_vals:
            f1_list = []
            for fold in range(5):
                val_idx = np.where(fold_ids == fold)[0]
                ref_idx = np.where(fold_ids != fold)[0]
                X_ref, y_ref = X_tr[ref_idx], y_tr[ref_idx]
                X_val, y_val = X_tr[val_idx], y_tr[val_idx]

                # Squared distance ranking is identical to Euclidean distance
                dists = np.sum((X_val[:, np.newaxis, :] - X_ref[np.newaxis, :, :]) ** 2, axis=2)
                preds = []
                for i in range(len(val_idx)):
                    s_idx = np.argsort(dists[i], kind="stable")
                    top_k = y_ref[s_idx[:k]]
                    preds.append(1 if np.sum(top_k) >= (k + 1) // 2 else 0)
                f1_list.append(confusion_metrics(y_val, np.array(preds))["f1"])
            cv_f1s.append(np.mean(f1_list))

        best_k = k_vals[int(np.argmax(cv_f1s))]

        with open(os.path.join("outputs", "part2_knn_cv.csv"), "w") as f:
            f.write("K,mean_cv_f1\n")
            for k, f1 in zip(k_vals, cv_f1s):
                f.write(f"{k},{f1:.4f}\n")

        # Refit KNN on full train
        te_dists = np.sum((X_te[:, np.newaxis, :] - X_tr[np.newaxis, :, :]) ** 2, axis=2)
        knn_preds = []
        for i in range(n_te):
            s_idx = np.argsort(te_dists[i], kind="stable")[:best_k]
            knn_preds.append(1 if np.sum(y_tr[s_idx]) >= (best_k + 1) // 2 else 0)
        knn_cm = confusion_metrics(y_te, np.array(knn_preds))

        # 2.8 Cost optimization
        p_tr = sigmoid(X_tr @ w + b)
        best_tau = 0.5
        min_train_cost = float("inf")
        for k in range(1, 20):
            tau = round(k / 20.0, 4)
            cm_tr = confusion_metrics(y_tr, (p_tr >= tau).astype(int))
            cost = 50 * cm_tr["fp"] + 400 * cm_tr["fn"]
            if cost < min_train_cost:
                min_train_cost = cost
                best_tau = tau

        def calc_test_cost(tau):
            cm = confusion_metrics(y_te, (p_te >= tau).astype(int))
            return 50 * cm["fp"] + 400 * cm["fn"]

        cost_test_at_tau = calc_test_cost(best_tau)
        cost_test_at_05 = calc_test_cost(0.5)
        cost_test_flag_nobody = 400 * int(np.sum(y_te == 1))

        with open(os.path.join("outputs", "part2_results.txt"), "w") as f:
            f.write(f"train_rows = {n_tr}\n")
            f.write(f"train_positives = {int(np.sum(y_tr == 1))}\n")
            f.write(f"test_rows = {n_te}\n")
            f.write(f"test_positives = {int(np.sum(y_te == 1))}\n")
            f.write(f"loss_epoch_0 = {losses[0]:.4f}\n")
            f.write(f"loss_epoch_1 = {losses[1]:.4f}\n")
            f.write(f"loss_epoch_10 = {losses[10]:.4f}\n")
            f.write(f"loss_epoch_100 = {losses[100]:.4f}\n")
            f.write(f"loss_epoch_1000 = {losses[1000]:.4f}\n")
            for i in range(7):
                f.write(f"weight_X{i + 1} = {w[i]:.4f}\n")
            f.write(f"bias = {b:.4f}\n")
            f.write(f"test_bce = {test_bce:.4f}\n")
            f.write(f"loss_epoch_200_eta_0.01 = {lr_losses[0.01]:.4f}\n")
            f.write(f"loss_epoch_200_eta_0.1 = {lr_losses[0.1]:.4f}\n")
            f.write(f"loss_epoch_200_eta_1.0 = {lr_losses[1.0]:.4f}\n")
            for i in range(7):
                f.write(f"odds_ratio_X{i + 1} = {odds_ratios[i]:.4f}\n")
            for prefix, cm in [("tau_0.5_", m_05), ("tau_0.3_", m_03)]:
                f.write(f"{prefix}tp = {cm['tp']}\n")
                f.write(f"{prefix}fp = {cm['fp']}\n")
                f.write(f"{prefix}tn = {cm['tn']}\n")
                f.write(f"{prefix}fn = {cm['fn']}\n")
                f.write(f"{prefix}accuracy = {cm['accuracy']:.4f}\n")
                f.write(f"{prefix}precision = {cm['precision']:.4f}\n")
                f.write(f"{prefix}recall = {cm['recall']:.4f}\n")
                f.write(f"{prefix}f1 = {cm['f1']:.4f}\n")
                f.write(f"{prefix}specificity = {cm['specificity']:.4f}\n")
            f.write(f"blind_accuracy = {blind_acc:.4f}\n")
            f.write(f"knn_best_k = {best_k}\n")
            f.write(f"knn_tp = {knn_cm['tp']}\n")
            f.write(f"knn_fp = {knn_cm['fp']}\n")
            f.write(f"knn_tn = {knn_cm['tn']}\n")
            f.write(f"knn_fn = {knn_cm['fn']}\n")
            f.write(f"knn_accuracy = {knn_cm['accuracy']:.4f}\n")
            f.write(f"knn_precision = {knn_cm['precision']:.4f}\n")
            f.write(f"knn_recall = {knn_cm['recall']:.4f}\n")
            f.write(f"knn_f1 = {knn_cm['f1']:.4f}\n")
            f.write(f"cost_tau = {best_tau:.4f}\n")
            f.write(f"cost_train_at_tau = {min_train_cost:.4f}\n")
            f.write(f"cost_test_at_tau = {cost_test_at_tau:.4f}\n")
            f.write(f"cost_test_at_0.5 = {cost_test_at_05:.4f}\n")
            f.write(f"cost_test_flag_nobody = {cost_test_flag_nobody:.4f}\n")

    if __name__ == "__main__":
        run()