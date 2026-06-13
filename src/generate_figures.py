from __future__ import annotations

import json
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
W5_DIR = os.path.join(PROJECT_ROOT, "reports", "Week5", "figs")
W10_DIR = os.path.join(PROJECT_ROOT, "reports", "Week10")
os.makedirs(W5_DIR, exist_ok=True)
os.makedirs(W10_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


# ---------------------------------------------------------------------------
# Week 5 figures
# ---------------------------------------------------------------------------

def gen_w5_figures():
    print("Generating Week 5 figures...")
    with open(os.path.join(ARTIFACTS_DIR, "models.pkl"), "rb") as f:
        models = pickle.load(f)

    pca = models["pca"]
    svd = models["svd"]

    # Fig W5-1: PCA cumulative variance
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    k90 = int(np.argmax(cum_var >= 0.90)) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cum_var) + 1), cum_var * 100, linewidth=2, color="steelblue")
    ax.axhline(90, ls="--", color="crimson", label="90% threshold")
    ax.axvline(k90, ls=":", color="orange", label=f"k={k90} components")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA on X_numeric: cumulative explained variance")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(W5_DIR, "fig_w5_01_pca_variance.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")

    # Fig W5-2: SVD singular value spectrum (text)
    sv = svd.singular_values_
    cum_sv2 = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    k50 = int(np.argmax(cum_sv2 >= 0.50)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(range(1, len(sv) + 1), sv, linewidth=2, color="seagreen")
    axes[0].set_xlabel("Component index")
    axes[0].set_ylabel("Singular value")
    axes[0].set_title("SVD on X_text: singular values")

    axes[1].plot(range(1, len(cum_sv2) + 1), cum_sv2 * 100, linewidth=2, color="darkorange")
    axes[1].axhline(50, ls="--", color="crimson", label="50% threshold")
    axes[1].axvline(k50, ls=":", color="steelblue", label=f"k={k50} components")
    axes[1].set_xlabel("Number of SVD components")
    axes[1].set_ylabel("Cumulative variance explained (%)")
    axes[1].set_title("SVD on X_text: cumulative variance")
    axes[1].legend()
    plt.tight_layout()
    out = os.path.join(W5_DIR, "fig_w5_02_svd_spectrum.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")

    # Fig W5-3: Feature matrix dimensions comparison
    matrices = {"X_numeric\n(dense)": (87806, 26), "X_categorical\n(sparse)": (87806, 5000), "X_text\n(sparse)": (87806, 5000)}
    names = list(matrices.keys())
    cols = [v[1] for v in matrices.values()]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, cols, color=["steelblue", "darkorange", "seagreen"], width=0.5)
    for bar, val in zip(bars, cols):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, f"{val:,}", ha="center", fontsize=10)
    ax.set_ylabel("Number of features (columns)")
    ax.set_title("Feature matrix dimensions — 87,806 games")
    plt.tight_layout()
    out = os.path.join(W5_DIR, "fig_w5_03_matrix_dims.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")

    return k90, k50


# ---------------------------------------------------------------------------
# Week 10 figures
# ---------------------------------------------------------------------------

def gen_w10_figures():
    print("Generating Week 10 figures...")
    with open(os.path.join(ARTIFACTS_DIR, "recommendation", "rec_metrics.json")) as f:
        metrics = json.load(f)

    ev = metrics["evaluation"]
    best_k = metrics["mf_best_k"]

    # Fig 1: HR@K comparison bar chart
    rows = [
        ("Popularity",       ev["popularity_baseline"]),
        ("Content",          ev["content_based_baseline"]),
        ("MF k=10",          ev["matrix_factorization"].get("k=10", {})),
        ("MF k=20",          ev["matrix_factorization"].get("k=20", {})),
        ("MF k=50",          ev["matrix_factorization"].get("k=50", {})),
    ]
    best_alpha, best_hres = max(ev["hybrid"].items(), key=lambda x: x[1].get("HR@10", 0))
    rows.append((f"Hybrid\n{best_alpha}", best_hres))

    names = [r[0] for r in rows]
    hr5  = [r[1].get("HR@5",  0) for r in rows]
    hr10 = [r[1].get("HR@10", 0) for r in rows]
    hr20 = [r[1].get("HR@20", 0) for r in rows]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, hr5,  w, label="HR@5",  color="steelblue")
    ax.bar(x,     hr10, w, label="HR@10", color="darkorange")
    ax.bar(x + w, hr20, w, label="HR@20", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Hit-Rate")
    ax.set_title("HR@K Comparison: Popularity vs Content vs MF vs Hybrid")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    plt.tight_layout()
    out = os.path.join(W10_DIR, "fig01_hr_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")

    # Fig 2: MF loss curve
    loss_history = metrics["mf_loss_history"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, marker="o", markersize=4, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Regularized training loss")
    ax.set_title(f"Matrix Factorization: SGD loss curve (k={best_k})")
    plt.tight_layout()
    out = os.path.join(W10_DIR, "fig02_mf_loss_curve.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")

    # Fig 3: MF k-sweep
    mf_sweep = ev["matrix_factorization"]
    k_labels = [int(k.split("=")[1]) for k in mf_sweep]
    hr10_vals = [mf_sweep[k]["HR@10"] for k in mf_sweep]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_labels, hr10_vals, marker="o", linewidth=2, color="seagreen")
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("HR@10")
    ax.set_title("MF: HR@10 vs latent dimension k")
    ax.set_xticks(k_labels)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    out = os.path.join(W10_DIR, "fig03_mf_k_sweep.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")

    # Fig 4: Hybrid alpha sweep
    hybrid = ev["hybrid"]
    alphas = [float(a.split("=")[1]) for a in hybrid]
    hr10_hybrid = [hybrid[a]["HR@10"] for a in hybrid]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, hr10_hybrid, marker="s", linewidth=2, color="darkorange", label="Hybrid")
    ax.axhline(ev["content_based_baseline"]["HR@10"], ls="--", color="steelblue", label="Content-only")
    ax.axhline(ev["matrix_factorization"][f"k={best_k}"]["HR@10"], ls="--", color="seagreen", label=f"MF k={best_k}")
    ax.set_xlabel("alpha (weight on content score)")
    ax.set_ylabel("HR@10")
    ax.set_title("Hybrid: HR@10 vs alpha (content weight)")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    out = os.path.join(W10_DIR, "fig04_hybrid_alpha_sweep.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    gen_w5_figures()
    gen_w10_figures()
    print("All figures generated.")
