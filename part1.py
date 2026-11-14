"""Part 1: Estimation, inference and regression baseline."""
import os
import math
import numpy as np


def run():
    # Load data
    data_path = os.path.join("data", "records.csv")
    raw = []
    with open(data_path, "r") as f:
        lines = [line.strip().split(",") for line in f if line.strip()]
    header = lines[0]
    rows = lines[1:]

    n = len(rows)
    # Columns: record_id(0), X1(1), X2(2), X3(3), X4(4), X5(5), X6(6), X7(7), outcome(8), value(9)
    val = np.array([float(r[9]) for r in rows], dtype=np.float64)
    x4 = np.array([float(r[4]) for r in rows], dtype=np.float64)

    # 1.1 Estimate with an interval
    mean_val = np.mean(val)
    sd_val = np.std(val, ddof=1)
    se_val = sd_val / math.sqrt(n)
    ci95_low = mean_val - 1.96 * se_val
    ci95_high = mean_val + 1.96 * se_val

    # 1.2 Test planning claim (mu = 8.0)
    mu_0 = 8.0
    z_stat = (mean_val - mu_0) / se_val
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # 1.3 Sample size
    E = 0.5
    n_req = math.ceil((1.96 * sd_val / E) ** 2)

    # 1.4 Simple regression by closed form: value on X4
    mean_x4 = np.mean(x4)
    cov_x4_val = np.sum((x4 - mean_x4) * (val - mean_val)) / (n - 1)
    var_x4 = np.sum((x4 - mean_x4) ** 2) / (n - 1)
    reg_slope = cov_x4_val / var_x4
    reg_intercept = mean_val - reg_slope * mean_x4

    fitted = reg_intercept + reg_slope * x4
    residuals = val - fitted
    tss = np.sum((val - mean_val) ** 2)
    ess = np.sum((fitted - mean_val) ** 2)
    rss = np.sum(residuals ** 2)
    tss_eq = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / n)
    mae = np.mean(np.abs(residuals))

    # 1.5 Fit same line by Gradient Descent
    z_x4 = (x4 - mean_x4) / sd_val_x4 if (sd_val_x4 := np.std(x4, ddof=1)) else x4
    m, c = 0.0, 0.0
    eta = 0.1
    gd_costs = {}
    for step in range(501):
        y_hat = m * z_x4 + c
        err = val - y_hat
        cost = np.mean(err ** 2)
        if step in [0, 1, 10, 100, 500]:
            gd_costs[step] = cost
        if step < 500:
            grad_m = - (2.0 / n) * np.sum(z_x4 * err)
            grad_c = - (2.0 / n) * np.sum(err)
            m -= eta * grad_m
            c -= eta * grad_c

    gd_slope = m / sd_val_x4
    gd_intercept = c - m * mean_x4 / sd_val_x4
    gd_slope_gap = abs(gd_slope - reg_slope)

    # 1.6 Heteroscedastic funnel & log remedy
    sort_idx = np.argsort(fitted, kind="stable")
    low_idx, up_idx = sort_idx[:400], sort_idx[400:]
    resid_sd_lower_raw = np.std(residuals[low_idx], ddof=1)
    resid_sd_upper_raw = np.std(residuals[up_idx], ddof=1)
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    ln_val = np.log(val)
    ln_x4 = np.log(x4)
    mean_ln_x4 = np.mean(ln_x4)
    mean_ln_val = np.mean(ln_val)
    cov_ln = np.sum((ln_x4 - mean_ln_x4) * (ln_val - mean_ln_val)) / (n - 1)
    var_ln = np.sum((ln_x4 - mean_ln_x4) ** 2) / (n - 1)
    loglog_slope = cov_ln / var_ln
    loglog_intercept = mean_ln_val - loglog_slope * mean_ln_x4

    loglog_fitted = loglog_intercept + loglog_slope * ln_x4
    loglog_residuals = ln_val - loglog_fitted
    tss_log = np.sum((ln_val - mean_ln_val) ** 2)
    rss_log = np.sum(loglog_residuals ** 2)
    loglog_r2 = 1.0 - rss_log / tss_log
    loglog_rmse = math.sqrt(rss_log / n)
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

    sort_idx_log = np.argsort(loglog_fitted, kind="stable")
    resid_sd_lower_log = np.std(loglog_residuals[sort_idx_log[:400]], ddof=1)
    resid_sd_upper_log = np.std(loglog_residuals[sort_idx_log[400:]], ddof=1)
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    # 1.7 Multiple regression by normal equation
    # ln(value) on 1, ln(X4), X1, X2, X3, X5, X6, X7
    X_mat = np.column_stack([
        np.ones(n),
        ln_x4,
        np.array([float(r[1]) for r in rows]),
        np.array([float(r[2]) for r in rows]),
        np.array([float(r[3]) for r in rows]),
        np.array([float(r[5]) for r in rows]),
        np.array([float(r[6]) for r in rows]),
        np.array([float(r[7]) for r in rows])
    ])
    beta = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ ln_val)
    mr_fitted = X_mat @ beta
    mr_rss = np.sum((ln_val - mr_fitted) ** 2)
    mr_r2 = 1.0 - mr_rss / tss_log
    p = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p - 1)

    os.makedirs("outputs", exist_ok=True)
    with open(os.path.join("outputs", "part1_results.txt"), "w") as f:
        f.write(f"n = {n}\n")
        f.write(f"mean_value = {mean_val:.4f}\n")
        f.write(f"sd_value = {sd_val:.4f}\n")
        f.write(f"se_value = {se_val:.4f}\n")
        f.write(f"ci95_low = {ci95_low:.4f}\n")
        f.write(f"ci95_high = {ci95_high:.4f}\n")
        f.write(f"z_stat = {z_stat:.4f}\n")
        f.write(f"reject_h0 = {reject_h0}\n")
        f.write(f"n_required = {n_req}\n")
        f.write(f"reg_slope = {reg_slope:.4f}\n")
        f.write(f"reg_intercept = {reg_intercept:.4f}\n")
        f.write(f"tss = {tss:.4f}\n")
        f.write(f"ess = {ess:.4f}\n")
        f.write(f"rss = {rss:.4f}\n")
        f.write(f"tss_equals_ess_plus_rss = {tss_eq}\n")
        f.write(f"r2 = {r2:.4f}\n")
        f.write(f"rmse = {rmse:.4f}\n")
        f.write(f"mae = {mae:.4f}\n")
        f.write(f"gd_J0 = {gd_costs[0]:.4f}\n")
        f.write(f"gd_J1 = {gd_costs[1]:.4f}\n")
        f.write(f"gd_J10 = {gd_costs[10]:.4f}\n")
        f.write(f"gd_J100 = {gd_costs[100]:.4f}\n")
        f.write(f"gd_J500 = {gd_costs[500]:.4f}\n")
        f.write(f"gd_slope = {gd_slope:.4f}\n")
        f.write(f"gd_intercept = {gd_intercept:.4f}\n")
        f.write(f"gd_slope_gap = {gd_slope_gap:.4f}\n")
        f.write(f"resid_sd_lower_raw = {resid_sd_lower_raw:.4f}\n")
        f.write(f"resid_sd_upper_raw = {resid_sd_upper_raw:.4f}\n")
        f.write(f"resid_sd_ratio_raw = {resid_sd_ratio_raw:.4f}\n")
        f.write(f"loglog_intercept = {loglog_intercept:.4f}\n")
        f.write(f"loglog_slope = {loglog_slope:.4f}\n")
        f.write(f"loglog_r2 = {loglog_r2:.4f}\n")
        f.write(f"loglog_rmse = {loglog_rmse:.4f}\n")
        f.write(f"resid_sd_lower_log = {resid_sd_lower_log:.4f}\n")
        f.write(f"resid_sd_upper_log = {resid_sd_upper_log:.4f}\n")
        f.write(f"resid_sd_ratio_log = {resid_sd_ratio_log:.4f}\n")
        for i in range(8):
            f.write(f"mr_beta_{i} = {beta[i]:.4f}\n")
        f.write(f"mr_r2 = {mr_r2:.4f}\n")
        f.write(f"mr_adj_r2 = {mr_adj_r2:.4f}\n")
        f.write(f"loglog_adj_r2 = {loglog_adj_r2:.4f}\n")


if __name__ == "__main__":
    run()