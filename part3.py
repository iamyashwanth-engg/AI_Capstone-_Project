"""Part 3: Segmentation and dimensionality reduction."""
import os
import csv
import math
import numpy as np


def run():
    data_path = os.path.join("data", "records.csv")
    with open(data_path, "r", newline="") as f:
        lines = [l.strip().split(",") for l in f if l.strip()]
    rows = lines[1:]
    n = len(rows)

    X_orig = np.array([[float(r[j]) for j in range(1, 6)] for r in rows], dtype=np.float64)
    outcomes = np.array([int(r[8]) for r in rows], dtype=np.int32)
    values = np.array([float(r[9]) for r in rows], dtype=np.float64)

    mean_vec = np.mean(X_orig, axis=0)
    std_vec = np.std(X_orig, axis=0, ddof=1)
    X_std = (X_orig - mean_vec) / std_vec

    # 3.1 Pinned K-means
    wcss_dict = {}
    k3_labels = None
    for K in [2, 3, 4, 5, 6]:
        centroids = np.array([X_std[i * (800 // K)] for i in range(K)], dtype=np.float64)
        labels = np.full(n, -1, dtype=int)
        while True:
            dists = np.sum((X_std[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for k in range(K):
                pts = X_std[labels == k]
                if len(pts) > 0:
                    centroids[k] = np.mean(pts, axis=0)

        wcss = np.sum(np.min(np.sum((X_std[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2), axis=1))
        wcss_dict[K] = wcss
        if K == 3:
            k3_labels = labels

    # 3.2 Elbow rule
    elbow_k = 3
    for K in [3, 4, 5]:
        drop_curr = wcss_dict[K - 1] - wcss_dict[K]
        drop_next = wcss_dict[K] - wcss_dict[K + 1]
        if drop_next < 0.5 * drop_curr:
            elbow_k = K
            break

    # 3.3 Segment profiling
    seg_means = [(np.mean(X_orig[k3_labels == k, 2]), k) for k in range(3)]
    seg_means.sort(key=lambda t: t[0])
    label_to_seg = {seg_means[0][1]: "A", seg_means[1][1]: "B", seg_means[2][1]: "C"}
    final_segments = [label_to_seg[lbl] for lbl in k3_labels]

    seg_info = {}
    for letter in ["A", "B", "C"]:
        idx = [i for i, s in enumerate(final_segments) if s == letter]
        sub_X = X_orig[idx]
        seg_info[letter] = {
            "size": len(idx),
            "means": np.mean(sub_X, axis=0),
            "outcome_rate": np.mean(outcomes[idx]),
            "mean_value": np.mean(values[idx])
        }

    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "part3_segments.csv"), "w") as f:
        f.write("segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value\n")
        for s in ["A", "B", "C"]:
            m = seg_info[s]["means"]
            f.write(
                f"{s},{seg_info[s]['size']},{m[0]:.4f},{m[1]:.4f},{m[2]:.4f},{m[3]:.4f},{m[4]:.4f},{seg_info[s]['outcome_rate']:.4f},{seg_info[s]['mean_value']:.4f}\n")

    # 3.4 Hierarchical clustering
    dist_matrix = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
    ])
    cohorts = [f"E{i + 1}" for i in range(6)]

    hier_rows = []
    clusters_at_cut = {}
    for linkage in ["single", "complete", "average"]:
        current_clusters = [[c] for c in cohorts]
        merges = []
        for step in range(5):
            min_dist = float("inf")
            best_pair = None
            for i in range(len(current_clusters)):
                for j in range(i + 1, len(current_clusters)):
                    ca, cb = current_clusters[i], current_clusters[j]
                    dists = [dist_matrix[cohorts.index(a), cohorts.index(b)] for a in ca for b in cb]
                    d = min(dists) if linkage == "single" else (max(dists) if linkage == "complete" else np.mean(dists))
                    if d < min_dist:
                        min_dist = d
                        best_pair = (i, j)
            i, j = best_pair
            ca, cb = current_clusters[i], current_clusters[j]
            min_a = min(int(x[1:]) for x in ca)
            min_b = min(int(x[1:]) for x in cb)
            c1_str = "+".join(sorted(ca, key=lambda x: int(x[1:])))
            c2_str = "+".join(sorted(cb, key=lambda x: int(x[1:])))
            if min_b < min_a:
                c1_str, c2_str = c2_str, c1_str
            merges.append((step + 1, c1_str, c2_str, min_dist))
            current_clusters = [c for idx, c in enumerate(current_clusters) if idx not in [i, j]] + [ca + cb]
            hier_rows.append((linkage, step + 1, c1_str, c2_str, min_dist))

        heights = [m[3] for m in merges]
        gaps = [heights[i + 1] - heights[i] for i in range(4)]
        merges_below_cut = int(np.argmax(gaps)) + 1
        clusters_at_cut[linkage] = 6 - merges_below_cut

    with open(os.path.join("outputs", "part3_hierarchical.csv"), "w") as f:
        f.write("linkage,merge_no,cluster_a,cluster_b,height\n")
        for r in hier_rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f}\n")

    # 3.5 PCA with sign normalization
    S = (1.0 / (n - 1)) * (X_std.T @ X_std)
    eigvals, eigvecs = np.linalg.eigh(S)

    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    # Flip signs so element of largest absolute value is positive
    for col in range(eigvecs.shape[1]):
        max_idx = np.argmax(np.abs(eigvecs[:, col]))
        if eigvecs[max_idx, col] < 0:
            eigvecs[:, col] *= -1.0

    eig_sum = np.sum(eigvals)
    exp_ratios = eigvals / eig_sum
    cum_ratios = np.cumsum(exp_ratios)
    comp_80 = int(np.where(cum_ratios >= 0.8)[0][0]) + 1

    W = eigvecs[:, :2]
    Z = X_std @ W

    mean_pc1_seg = {letter: np.mean(Z[[i for i, s in enumerate(final_segments) if s == letter], 0]) for letter in
                    ["A", "B", "C"]}

    with open(os.path.join("outputs", "part3_results.txt"), "w") as f:
        for K in [2, 3, 4, 5, 6]:
            f.write(f"wcss_k{K} = {wcss_dict[K]:.4f}\n")
        f.write(f"elbow_k = {elbow_k}\n")
        f.write(f"size_segment_A = {seg_info['A']['size']}\n")
        f.write(f"size_segment_B = {seg_info['B']['size']}\n")
        f.write(f"size_segment_C = {seg_info['C']['size']}\n")
        f.write(f"single_clusters_at_cut = {clusters_at_cut['single']}\n")
        f.write(f"complete_clusters_at_cut = {clusters_at_cut['complete']}\n")
        f.write(f"average_clusters_at_cut = {clusters_at_cut['average']}\n")
        f.write(f"eigenvalue_sum = {eig_sum:.4f}\n")
        for i in range(5):
            f.write(f"eigenvalue_{i + 1} = {eigvals[i]:.4f}\n")
        for i in range(5):
            f.write(f"explained_ratio_{i + 1} = {exp_ratios[i]:.4f}\n")
        for i in range(5):
            f.write(f"cumulative_ratio_{i + 1} = {cum_ratios[i]:.4f}\n")
        f.write(f"components_for_80pct = {comp_80}\n")
        for i in range(5):
            f.write(f"pc1_loading_X{i + 1} = {eigvecs[i, 0]:.4f}\n")
        for i in range(5):
            f.write(f"pc2_loading_X{i + 1} = {eigvecs[i, 1]:.4f}\n")
        f.write(f"row1_pc1 = {Z[0, 0]:.4f}\n")
        f.write(f"row1_pc2 = {Z[0, 1]:.4f}\n")
        f.write(f"row2_pc1 = {Z[1, 0]:.4f}\n")
        f.write(f"row2_pc2 = {Z[1, 1]:.4f}\n")
        f.write(f"row3_pc1 = {Z[2, 0]:.4f}\n")
        f.write(f"row3_pc2 = {Z[2, 1]:.4f}\n")
        f.write(f"mean_pc1_segment_A = {mean_pc1_seg['A']:.4f}\n")
        f.write(f"mean_pc1_segment_B = {mean_pc1_seg['B']:.4f}\n")
        f.write(f"mean_pc1_segment_C = {mean_pc1_seg['C']:.4f}\n")


if __name__ == "__main__":
    run()