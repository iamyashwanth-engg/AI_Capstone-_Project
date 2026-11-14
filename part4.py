"""Part 4: Model-choice and architecture memo."""
import os
import math
import numpy as np
from shared_helper import load_and_split, confusion_metrics


def entropy(pos, total):
    if total == 0 or pos == 0 or pos == total:
        return 0.0
    p1 = pos / total
    p0 = 1.0 - p1
    return - (p1 * math.log2(p1) + p0 * math.log2(p0))


def gini(pos, total):
    if total == 0:
        return 0.0
    p1 = pos / total
    p0 = 1.0 - p1
    return 1.0 - (p1 ** 2 + p0 ** 2)


def run():
    sp = load_and_split()
    X_tr_orig = sp["X_train_orig"]
    y_tr = sp["y_train"]
    X_te_orig = sp["X_test_orig"]
    X_tr_std = sp["X_train_std"]
    X_te_std = sp["X_test_std"]
    y_te = sp["y_test"]
    n_tr = len(y_tr)
    n_te = len(y_te)

    # 4.1 Split purity
    pos_root = int(np.sum(y_tr == 1))
    root_ent = entropy(pos_root, n_tr)
    root_g = gini(pos_root, n_tr)

    splits = [
        ("S1", X_tr_orig[:, 5] == 1.0, X_te_orig[:, 5] == 1.0),
        ("S2", X_tr_orig[:, 6] == 1.0, X_te_orig[:, 6] == 1.0),
        ("S3", X_tr_orig[:, 2] > 0.5, X_te_orig[:, 2] > 0.5),
        ("S4", X_tr_orig[:, 0] <= 24.0, X_te_orig[:, 0] <= 24.0)
    ]

    split_records = []
    for name, mask_tr, mask_te in splits:
        n_y = int(np.sum(mask_tr))
        p_y = int(np.sum(y_tr[mask_tr] == 1))
        n_n = n_tr - n_y
        p_n = int(np.sum(y_tr[~mask_tr] == 1))

        ent_y = entropy(p_y, n_y)
        ent_n = entropy(p_n, n_n)
        inf_gain = root_ent - ((n_y / n_tr) * ent_y + (n_n / n_tr) * ent_n)

        gin_y = gini(p_y, n_y)
        gin_n = gini(p_n, n_n)
        gin_gain = root_g - ((n_y / n_tr) * gin_y + (n_n / n_tr) * gin_n)
        split_records.append((name, n_y, p_y, n_n, p_n, inf_gain, gin_gain, mask_tr, mask_te))

    best_inf = max(split_records, key=lambda x: x[5])
    best_gin = max(split_records, key=lambda x: x[6])

    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "part4_splits.csv"), "w") as f:
        f.write("split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain\n")
        for r in split_records:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f},{r[6]:.4f}\n")

    # 4.2 Stump & ensembles
    stump_rate_yes = best_inf[2] / best_inf[1]
    stump_rate_no = best_inf[4] / best_inf[3]
    stump_probs_te = np.where(best_inf[8], stump_rate_yes, stump_rate_no)
    stump_preds = (stump_probs_te >= 0.3).astype(int)

    # Logistic
    w, b = np.zeros(7), 0.0
    for _ in range(1000):
        p = 1.0 / (1.0 + np.exp(-np.clip(X_tr_std @ w + b, -30.0, 30.0)))
        w -= 0.1 * (1.0 / n_tr) * (X_tr_std.T @ (p - y_tr))
        b -= 0.1 * (1.0 / n_tr) * np.sum(p - y_tr)
    log_probs_te = 1.0 / (1.0 + np.exp(-np.clip(X_te_std @ w + b, -30.0, 30.0)))
    log_preds = (log_probs_te >= 0.3).astype(int)

    # KNN
    k_vals = list(range(1, 23, 2))
    cv_scores = []
    f_ids = np.array([i % 5 for i in range(n_tr)])
    for k in k_vals:
        f1_list = []
        for f in range(5):
            val_idx = np.where(f_ids == f)[0]
            ref_idx = np.where(f_ids != f)[0]
            d = np.sum((X_tr_std[val_idx, np.newaxis, :] - X_tr_std[ref_idx][np.newaxis, :, :]) ** 2, axis=2)
            preds = [1 if np.sum(y_tr[ref_idx][np.argsort(d[i], kind="stable")[:k]]) >= (k + 1) // 2 else 0 for i in
                     range(len(val_idx))]
            f1_list.append(confusion_metrics(y_tr[val_idx], np.array(preds))["f1"])
        cv_scores.append(np.mean(f1_list))
    best_k = k_vals[int(np.argmax(cv_scores))]

    te_d = np.sum((X_te_std[:, np.newaxis, :] - X_tr_std[np.newaxis, :, :]) ** 2, axis=2)
    knn_probs_te = np.zeros(n_te)
    for i in range(n_te):
        s_idx = np.argsort(te_d[i], kind="stable")[:best_k]
        knn_probs_te[i] = np.mean(y_tr[s_idx])
    knn_preds = (knn_probs_te >= 0.5).astype(int)

    # Ensembles
    hard_vote_preds = ((log_preds + knn_preds + stump_preds) >= 2).astype(int)
    soft_vote_probs = (log_probs_te + knn_probs_te + stump_probs_te) / 3.0
    soft_vote_preds = (soft_vote_probs >= 0.3).astype(int)

    models = [
        ("logistic", log_preds),
        ("knn", knn_preds),
        ("stump", stump_preds),
        ("hard_vote", hard_vote_preds),
        ("soft_vote", soft_vote_preds)
    ]
    with open(os.path.join("outputs", "part4_voting.csv"), "w") as f:
        f.write("model,tp,fp,tn,fn,precision,recall,f1\n")
        for m_name, preds in models:
            cm = confusion_metrics(y_te, preds)
            f.write(
                f"{m_name},{cm['tp']},{cm['fp']},{cm['tn']},{cm['fn']},{cm['precision']:.4f},{cm['recall']:.4f},{cm['f1']:.4f}\n")

    # 4.3 AdaBoost
    y_b = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

    w_b = np.full(8, 0.125)
    adaboost_rows = []

    err1 = np.sum(w_b[h1 != y_b])
    alpha1 = 0.5 * math.log((1.0 - err1) / err1)
    unnorm_w1 = w_b * np.exp(- alpha1 * y_b * h1)
    w_after1 = unnorm_w1 / np.sum(unnorm_w1)
    for i in range(8):
        adaboost_rows.append((1, i + 1, w_b[i], unnorm_w1[i], w_after1[i]))

    err2 = np.sum(w_after1[h2 != y_b])
    alpha2 = 0.5 * math.log((1.0 - err2) / err2)
    unnorm_w2 = w_after1 * np.exp(- alpha2 * y_b * h2)
    w_after2 = unnorm_w2 / np.sum(unnorm_w2)
    for i in range(8):
        adaboost_rows.append((2, i + 1, w_after1[i], unnorm_w2[i], w_after2[i]))

    final_pred = np.sign(alpha1 * h1 + alpha2 * h2)
    ens_acc = np.mean(final_pred == y_b)

    with open(os.path.join("outputs", "part4_adaboost.csv"), "w") as f:
        f.write("round,point,weight_before,weight_unnormalised,weight_after\n")
        for r in adaboost_rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")

    # 4.4 OOB fraction
    oob_frac = (1.0 - 1.0 / 560.0) ** 560

    # 4.5 Architectures
    p_mlp_h1 = 784 * 16 + 16
    p_mlp_h2 = 16 * 16 + 16
    p_mlp_out = 16 * 10 + 10
    mlp_tot = p_mlp_h1 + p_mlp_h2 + p_mlp_out

    p_cnn_c1 = (5 * 5 * 1 + 1) * 6
    p_cnn_c2 = (5 * 5 * 6 + 1) * 16
    p_cnn_d1 = (5 * 5 * 16) * 32 + 32
    p_cnn_d2 = 32 * 10 + 10
    cnn_tot = p_cnn_c1 + p_cnn_c2 + p_cnn_d1 + p_cnn_d2

    arch_rows = [
        ("mlp_hidden1", p_mlp_h1),
        ("mlp_hidden2", p_mlp_h2),
        ("mlp_output", p_mlp_out),
        ("cnn_conv1", p_cnn_c1),
        ("cnn_conv2", p_cnn_c2),
        ("cnn_dense1", p_cnn_d1),
        ("cnn_dense2", p_cnn_d2)
    ]
    with open(os.path.join("outputs", "part4_architecture.csv"), "w") as f:
        f.write("layer,parameters\n")
        for r in arch_rows:
            f.write(f"{r[0]},{r[1]}\n")

    fc_params = 128 * 128 * 3 * 1024 + 1024
    conv_params = (3 * 3 * 3 + 1) * 64
    rnn_total = 128 * 64 + 128 * 128 + 128 + 16 * 128 + 16

    with open(os.path.join("outputs", "part4_results.txt"), "w") as f:
        f.write(f"root_entropy = {root_ent:.4f}\n")
        f.write(f"root_gini = {root_g:.4f}\n")
        f.write(f"best_split_by_info_gain = {best_inf[0]}\n")
        f.write(f"best_split_by_gini_gain = {best_gin[0]}\n")
        f.write(f"stump_rate_yes = {stump_rate_yes:.4f}\n")
        f.write(f"stump_rate_no = {stump_rate_no:.4f}\n")
        f.write(f"round1_error = {err1:.4f}\n")
        f.write(f"round1_alpha = {alpha1:.4f}\n")
        f.write(f"round2_error = {err2:.4f}\n")
        f.write(f"round2_alpha = {alpha2:.4f}\n")
        f.write(f"ensemble_accuracy = {ens_acc:.4f}\n")
        f.write(f"oob_fraction = {oob_frac:.4f}\n")
        f.write(f"mlp_total = {mlp_tot}\n")
        f.write(f"cnn_total = {cnn_tot}\n")
        f.write(f"fc_layer_params = {fc_params}\n")
        f.write(f"conv_layer_params = {conv_params}\n")
        f.write(f"rnn_total = {rnn_total}\n")


if __name__ == "__main__":
    run()