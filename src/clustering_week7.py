"""
Week 7 - Prototype-Based Clustering Pipeline.

Reproduces the K-means experiments reported in:
    reports/Week7/Technical_report_G6_W7.pdf

Pipeline stages
---------------
1.  Load game catalog; filter to tagged games (~87,806 titles).
2.  Build a tags-only binary matrix (genres + categories + tags, min_freq=30 -> ~493 dims).
3.  Lloyd's manual descent verification (5 seeds, K=6, PCA-numeric 10D).
4.  Ablation: raw-sparse vs SVD-only vs SVD+L2 (silhouette comparison).
5.  K-sweep K in {4, 6, 8, 10} on the chosen SVD+L2 representation.
6.  Final K=6 clustering, cluster profiles by tag lift, ARI stability (5 seeds).

Artifacts written to artifacts/clustering/
------------------------------------------
    cluster_labels_k6.npy     - integer cluster label per tagged game
    cluster_appids.npy        - AppID array aligned to the labels
    clustering_metrics.json   - all numeric results (Lloyd's, ablation, sweep, ARI)
    cluster_profiles.json     - top-5 tags by lift per cluster

Run from project root:
    python src/clustering_week7.py
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer, normalize


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "catalog", "games_may2024_full.csv")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "clustering")

CLUSTERING_COLS = ["genres", "categories", "tags"]
MIN_FREQ = 30
SVD_COMPONENTS = 50
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_multilabel(value) -> list[str]:
    if pd.isna(value):
        return []
    text = re.sub(r"[\"'\[\]{}]", "", str(value))
    parts = re.split(r"[;,]", text)
    return [p.strip() for p in parts if p.strip() and not p.strip().isdigit()]


def load_tagged_catalog() -> pd.DataFrame:
    df = pd.read_csv(CATALOG_PATH)
    has_tags = df["tags"].notna() & (df["tags"].astype(str).str.strip().str.len() > 2)
    df = df[has_tags].reset_index(drop=True)
    print(f"Tagged games: {len(df):,}")
    return df


def build_tag_matrix(df: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str], list[list[str]]]:
    """Binary matrix on genres+categories+tags only, min_freq=MIN_FREQ."""
    from collections import Counter
    rows: list[list[str]] = []
    counts: Counter = Counter()
    for _, row in df[CLUSTERING_COLS].iterrows():
        tokens = []
        for col in CLUSTERING_COLS:
            for val in normalize_multilabel(row[col]):
                tokens.append(val.lower())
        rows.append(tokens)
        counts.update(tokens)

    allowed = {t for t, c in counts.items() if c >= MIN_FREQ}
    rows_filtered = [[t for t in r if t in allowed] for r in rows]
    allowed_sorted = sorted(allowed)
    print(f"Tag dimensions after min_freq={MIN_FREQ}: {len(allowed_sorted)}")

    mlb = MultiLabelBinarizer(classes=allowed_sorted, sparse_output=True)
    X = mlb.fit_transform(rows_filtered).tocsr()
    return X, allowed_sorted, rows_filtered


def build_numeric_features(df: pd.DataFrame) -> np.ndarray:
    """Numeric block for Lloyd's demo (same 23 columns as feature_engineering.py)."""
    from sklearn.preprocessing import StandardScaler
    numeric_cols = [
        "required_age", "price", "dlc_count", "metacritic_score",
        "achievements", "recommendations", "user_score", "score_rank",
        "positive", "negative", "estimated_owners", "average_playtime_forever",
        "average_playtime_2weeks", "median_playtime_forever", "median_playtime_2weeks",
        "peak_ccu", "pct_pos_total", "num_reviews_total", "pct_pos_recent",
        "num_reviews_recent",
    ]
    for col in ["windows", "mac", "linux"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        numeric_cols.append(col)

    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    num_df = num_df.replace([np.inf, -np.inf], np.nan)
    num_df = num_df.fillna(num_df.median(numeric_only=True)).fillna(0)
    lower, upper = num_df.quantile(0.01), num_df.quantile(0.99)
    num_df = num_df.clip(lower, upper, axis=1)
    return StandardScaler().fit_transform(num_df.values)


# ---------------------------------------------------------------------------
# Stage 3 -- Lloyd's manual descent verification
# ---------------------------------------------------------------------------

def _sse(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    total = 0.0
    for k in range(centroids.shape[0]):
        pts = X[labels == k]
        if len(pts):
            total += float(np.sum((pts - centroids[k]) ** 2))
    return total


def lloyds_one_iteration(
    X: np.ndarray, K: int, seed: int
) -> dict:
    """One assignment+update step from a random init; returns J_before and J_after."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), K, replace=False)
    centroids = X[idx].copy()

    # Assignment step (J before centroid update)
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    j_before = _sse(X, labels, centroids)

    # Update step
    new_centroids = np.array([
        X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
        for k in range(K)
    ])

    # Re-assign with new centroids (J after)
    dists2 = np.linalg.norm(X[:, None, :] - new_centroids[None, :, :], axis=2)
    labels2 = np.argmin(dists2, axis=1)
    j_after = _sse(X, labels2, new_centroids)

    return {
        "seed": seed,
        "J_before": round(j_before, 1),
        "J_after": round(j_after, 1),
        "delta_J": round(j_after - j_before, 0),
        "descent_ok": bool(j_after <= j_before),
    }


def run_lloyds_demo(X_numeric: np.ndarray) -> list[dict]:
    print("\n--- Stage 3: Lloyd's manual descent (5 seeds, K=6, PCA 10D) ---")
    pca = PCA(n_components=10, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_numeric)
    results = []
    for seed in range(1, 6):
        r = lloyds_one_iteration(X_pca, K=6, seed=seed)
        results.append(r)
        status = "Yes" if r["descent_ok"] else "NO"
        print(f"  Seed {seed}: J_before={r['J_before']:,.1f}  J_after={r['J_after']:,.1f}  "
              f"dJ={r['delta_J']:,.0f}  Descent: {status}")
    return results


# ---------------------------------------------------------------------------
# Stage 4 -- Ablation study
# ---------------------------------------------------------------------------

def run_ablation(X_sparse: sparse.csr_matrix, tag_dims: int) -> list[dict]:
    print("\n--- Stage 4: Ablation (raw sparse / SVD-only / SVD+L2) ---")
    K = 6
    results = []

    variants = [
        ("Raw sparse (binary tags)", X_sparse),
    ]

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    X_svd = svd.fit_transform(X_sparse)
    variants.append(("SVD only (50D dense)", X_svd))

    X_l2 = normalize(X_svd, norm="l2")
    variants.append(("SVD + L2 (chosen)", X_l2))

    for name, X_var in variants:
        km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE, init="k-means++")
        labels = km.fit_predict(X_var)
        sil = float(silhouette_score(X_var, labels, sample_size=10000, random_state=RANDOM_STATE))
        sizes = np.bincount(labels)
        largest_share = float(sizes.max() / len(labels))
        results.append({
            "variant": name,
            "silhouette": round(sil, 3),
            "largest_cluster_share_pct": round(largest_share * 100, 1),
        })
        print(f"  {name:<30}  silhouette={sil:.3f}  largest={largest_share:.1%}")

    return results


# ---------------------------------------------------------------------------
# Stage 5 -- K-sweep
# ---------------------------------------------------------------------------

def run_k_sweep(X_l2: np.ndarray) -> list[dict]:
    print("\n--- Stage 5: K-sweep K in {4, 6, 8, 10} ---")
    results = []
    for K in [4, 6, 8, 10]:
        km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE, init="k-means++")
        labels = km.fit_predict(X_l2)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X_l2, labels, sample_size=10000, random_state=RANDOM_STATE))
        sizes = np.bincount(labels)
        results.append({
            "K": K,
            "inertia": round(inertia, 0),
            "silhouette": round(sil, 4),
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
        })
        print(f"  K={K}  inertia={inertia:,.0f}  silhouette={sil:.4f}  "
              f"min={sizes.min():,}  max={sizes.max():,}")
    return results


# ---------------------------------------------------------------------------
# Stage 6 -- Final clustering K=6, profiles, ARI
# ---------------------------------------------------------------------------

def run_final_clustering(
    X_l2: np.ndarray, tag_names: list[str], df: pd.DataFrame
) -> tuple[np.ndarray, dict]:
    print("\n--- Stage 6a: Final K=6 clustering ---")
    K = 6
    km = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE, init="k-means++")
    labels = km.fit_predict(X_l2)
    inertia = float(km.inertia_)
    sil = float(silhouette_score(X_l2, labels, sample_size=10000, random_state=RANDOM_STATE))
    sizes = np.bincount(labels)
    print(f"  Inertia={inertia:,.0f}  Silhouette={sil:.4f}")
    for k in range(K):
        print(f"  C{k}: {sizes[k]:,} games ({sizes[k]/len(labels):.1%})")
    return labels, {"inertia": round(inertia, 0), "silhouette": round(sil, 4),
                    "cluster_sizes": sizes.tolist()}


def compute_cluster_profiles(
    labels: np.ndarray, tag_names: list[str], X_sparse: sparse.csr_matrix
) -> list[dict]:
    """Top-5 tags per cluster by lift = P(tag|cluster) / P(tag)."""
    print("\n--- Stage 6b: Cluster profiles by lift ---")
    n = len(labels)
    global_freq = np.asarray(X_sparse.mean(axis=0)).flatten()
    profiles = []
    for k in range(labels.max() + 1):
        mask = labels == k
        cluster_freq = np.asarray(X_sparse[mask].mean(axis=0)).flatten()
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(global_freq > 0, cluster_freq / global_freq, 0.0)
        top_idx = np.argsort(lift)[::-1][:5]
        top_tags = [
            {"tag": tag_names[i], "lift": round(float(lift[i]), 2),
             "cluster_freq": round(float(cluster_freq[i]), 3)}
            for i in top_idx
        ]
        profiles.append({"cluster": k, "size": int(mask.sum()), "top_tags_by_lift": top_tags})
        tag_str = ", ".join(f"{t['tag']} ({t['lift']}x)" for t in top_tags)
        print(f"  C{k} ({mask.sum():,} games): {tag_str}")
    return profiles


def run_ari_stability(X_l2: np.ndarray, n_seeds: int = 5) -> dict:
    """ARI between all pairs of K=6 runs across different seeds."""
    print("\n--- Stage 6c: ARI stability (5 seeds) ---")
    K = 6
    all_labels = []
    for seed in range(n_seeds):
        km = KMeans(n_clusters=K, n_init=10, random_state=seed, init="k-means++")
        all_labels.append(km.fit_predict(X_l2))

    ari_scores = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            ari_scores.append(float(adjusted_rand_score(all_labels[i], all_labels[j])))

    mean_ari = float(np.mean(ari_scores))
    min_ari = float(np.min(ari_scores))
    print(f"  Mean off-diagonal ARI={mean_ari:.4f}  Min ARI={min_ari:.4f}")
    return {"mean_ari": round(mean_ari, 4), "min_ari": round(min_ari, 4),
            "n_seed_pairs": len(ari_scores), "all_ari_scores": [round(a, 4) for a in ari_scores]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Load data
    print("Loading catalog...")
    df = load_tagged_catalog()
    appids = df["AppID"].astype(str).values

    # Build representations
    print("\nBuilding tag matrix...")
    X_sparse, tag_names, _ = build_tag_matrix(df)
    print(f"Tag matrix: {X_sparse.shape}")

    print("\nBuilding numeric features...")
    X_numeric = build_numeric_features(df)
    print(f"Numeric matrix: {X_numeric.shape}")

    # SVD + L2 (reused across stages 4, 5, 6)
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    X_svd = svd.fit_transform(X_sparse)
    X_l2 = normalize(X_svd, norm="l2")

    # Run all stages
    lloyds_results = run_lloyds_demo(X_numeric)
    ablation_results = run_ablation(X_sparse, len(tag_names))
    sweep_results = run_k_sweep(X_l2)
    labels, final_metrics = run_final_clustering(X_l2, tag_names, df)
    profiles = compute_cluster_profiles(labels, tag_names, X_sparse)
    ari_results = run_ari_stability(X_l2)

    # Save artifacts
    np.save(os.path.join(ARTIFACTS_DIR, "cluster_labels_k6.npy"), labels)
    np.save(os.path.join(ARTIFACTS_DIR, "cluster_appids.npy"), appids)

    metrics = {
        "n_games": int(len(df)),
        "n_tag_dimensions": len(tag_names),
        "svd_components": SVD_COMPONENTS,
        "lloyds_demo": lloyds_results,
        "ablation": ablation_results,
        "k_sweep": sweep_results,
        "final_k6": final_metrics,
        "ari_stability": ari_results,
    }
    with open(os.path.join(ARTIFACTS_DIR, "clustering_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(ARTIFACTS_DIR, "cluster_profiles.json"), "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    run_pipeline()
