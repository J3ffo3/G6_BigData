"""
Week 10 - Recommendation, Ranking, and Predictive Decision Engine.

Systems implemented
-------------------
1.  Popularity baseline     — rank games by aggregate recommendation count from catalog.
2.  Content-based baseline  — cosine similarity in SVD(TF-IDF + tags) space (50D).
3.  Matrix Factorization    — SGD latent-factor model on the user-game binary matrix.
4.  Hybrid recommender      — alpha * content_score + (1-alpha) * MF_score, alpha sweep.

Evaluation protocol
-------------------
- Interaction matrix : users x games, value = 1 when voted_up = True.
- Qualifying users   : users with >= MIN_INTERACTIONS positive interactions.
- Train/test split   : hold-out 1 randomly chosen positive item per qualifying user.
- Candidate pool     : all games the user has NOT interacted with in train.
- Metric             : Hit-Rate@K (HR@K) — fraction of users whose held-out item
                       appears in their top-K list.
- Primary metric     : HR@10.
- Hyperparameter sweeps: MF latent dim k in {10, 20, 50}; hybrid alpha in {0.1, 0.3, 0.5, 0.7, 0.9}.

Task framing
------------
This is a RECOMMENDATION task: given a user's positive interaction history, predict
which unseen game they would also endorse. The candidate pool is implicit (all unseen
games), and correctness is defined as recovering the held-out item in the top-K list.

Data alignment
--------------
All three systems operate on the same 500-game universe (V1 subset from steam_v1.parquet).
AppID is the common join key across catalog metadata and interaction records.
No hidden cross-catalog mapping is required — alignment is exact by construction.

Artifacts written to artifacts/recommendation/
----------------------------------------------
    rec_metrics.json   — all evaluation results (HR@K, parameter sweeps)
    mf_P.npy           — user latent factors (best MF configuration)
    mf_Q.npy           — item latent factors (best MF configuration)
    user_index.npy     — user IDs aligned to rows of mf_P
    item_index.npy     — game AppIDs aligned to rows of mf_Q
    content_matrix.npy — game content vectors (SVD 50D), aligned to item_index

Run from project root:
    python src/recommendation.py
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_FULL = os.path.join(PROJECT_ROOT, "data", "processed", "steam_full.parquet")
PARQUET_V1 = os.path.join(PROJECT_ROOT, "data", "processed", "steam_v1.parquet")
CATALOG_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "catalog", "games_may2024_full.csv")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "recommendation")

RANDOM_STATE = 42
MIN_INTERACTIONS = 2       # users need >= this many positives to qualify for hold-out
EVAL_K_VALUES = [5, 10, 20]
MF_K_VALUES = [10, 20, 50]
HYBRID_ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]
CONTENT_SVD_DIM = 50
MF_LR = 0.03
MF_REG = 0.01
MF_EPOCHS = 30


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_interactions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load interaction parquet and catalog CSV."""
    parquet_path = PARQUET_FULL if os.path.exists(PARQUET_FULL) else PARQUET_V1
    print(f"Loading interactions from: {os.path.basename(parquet_path)}")
    interactions = pd.read_parquet(parquet_path)
    interactions["AppID"] = interactions["AppID"].astype(str)
    interactions["author_steamid"] = interactions["author_steamid"].astype(str)
    interactions["voted_up"] = interactions["voted_up"].astype(int)

    print(f"Loading catalog from: {os.path.basename(CATALOG_PATH)}")
    catalog = pd.read_csv(
        CATALOG_PATH,
        usecols=["AppID", "name", "tags", "genres", "categories",
                 "short_description", "recommendations"],
    )
    catalog["AppID"] = catalog["AppID"].astype(str)
    return interactions, catalog


def build_interaction_universe(
    interactions: pd.DataFrame, catalog: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Filter to positive interactions; align to games present in both catalog and parquet.
    Returns:
        pos_df   — positive interactions (user, game)
        user_ids — sorted list of unique user IDs
        game_ids — sorted list of unique game AppIDs
    """
    pos = interactions[interactions["voted_up"] == 1][["author_steamid", "AppID"]].copy()
    pos = pos.drop_duplicates()

    # Keep only games that also appear in the catalog
    catalog_appids = set(catalog["AppID"].unique())
    pos = pos[pos["AppID"].isin(catalog_appids)]

    game_ids = sorted(pos["AppID"].unique())
    user_ids = sorted(pos["author_steamid"].unique())
    print(f"Positive interactions: {len(pos):,}")
    print(f"Unique users: {len(user_ids):,} | Unique games: {len(game_ids):,}")
    return pos, user_ids, game_ids


def train_test_split(
    pos_df: pd.DataFrame, user_ids: list[str], rng: np.random.Generator
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Hold out 1 positive item per qualifying user (>= MIN_INTERACTIONS).
    Returns:
        train_dict  : {user_id: [list of train game AppIDs]}
        held_out    : {user_id: held-out game AppID}
    """
    user_games = pos_df.groupby("author_steamid")["AppID"].apply(list).to_dict()
    train_dict: dict[str, list[str]] = {}
    held_out: dict[str, str] = {}

    for uid in user_ids:
        games = user_games.get(uid, [])
        if len(games) >= MIN_INTERACTIONS:
            idx = int(rng.integers(0, len(games)))
            held_out[uid] = games[idx]
            train_dict[uid] = [g for i, g in enumerate(games) if i != idx]

    print(f"Qualifying users (>={MIN_INTERACTIONS} positive): {len(held_out):,}")
    return train_dict, held_out


def build_sparse_matrix(
    train_dict: dict[str, list[str]],
    user_ids: list[str],
    game_ids: list[str],
) -> sparse.csr_matrix:
    """Build user x game binary sparse matrix from train interactions."""
    user_idx = {u: i for i, u in enumerate(user_ids)}
    game_idx = {g: i for i, g in enumerate(game_ids)}

    rows, cols = [], []
    for uid, games in train_dict.items():
        ui = user_idx[uid]
        for gid in games:
            if gid in game_idx:
                rows.append(ui)
                cols.append(game_idx[gid])

    data = np.ones(len(rows), dtype=np.float32)
    R = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(game_ids)),
        dtype=np.float32,
    )
    density = R.nnz / (R.shape[0] * R.shape[1])
    print(f"Interaction matrix: {R.shape}  density={density:.4%}")
    return R


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def hit_rate_at_k(
    scores_matrix: np.ndarray,
    train_matrix: sparse.csr_matrix,
    held_out: dict[str, str],
    user_ids: list[str],
    game_ids: list[str],
    k: int,
) -> float:
    """
    HR@K: fraction of qualifying users whose held-out item is in their top-K.
    Observed train interactions are masked to -inf before ranking.
    """
    user_idx = {u: i for i, u in enumerate(user_ids)}
    game_idx = {g: i for i, g in enumerate(game_ids)}

    hits = 0
    tested = 0
    for uid, held_game in held_out.items():
        if held_game not in game_idx:
            continue
        ui = user_idx[uid]
        scores = scores_matrix[ui].copy()
        # Mask train items
        train_items = train_matrix[ui].indices
        scores[train_items] = -np.inf
        top_k = np.argpartition(scores, -k)[-k:]
        hits += int(game_idx[held_game] in top_k)
        tested += 1

    return hits / tested if tested > 0 else 0.0


# ---------------------------------------------------------------------------
# Baseline 1 — Popularity
# ---------------------------------------------------------------------------

def popularity_baseline(
    catalog: pd.DataFrame, game_ids: list[str]
) -> np.ndarray:
    """
    Score vector for all games based on catalog `recommendations` count.
    Returns a 1D array of shape (n_games,), same order as game_ids.
    """
    game_idx = {g: i for i, g in enumerate(game_ids)}
    pop_map = catalog.set_index("AppID")["recommendations"].fillna(0).to_dict()
    scores = np.array([float(pop_map.get(gid, 0)) for gid in game_ids], dtype=float)
    # Broadcast as a (1, n_games) matrix for uniform interface with hit_rate_at_k
    return np.tile(scores, (1, 1))   # shape (1, n_games) — same score for every user


def eval_popularity(
    catalog: pd.DataFrame,
    train_matrix: sparse.csr_matrix,
    held_out: dict[str, str],
    user_ids: list[str],
    game_ids: list[str],
) -> dict:
    print("\n--- Baseline 1: Popularity ---")
    pop_scores_1d = popularity_baseline(catalog, game_ids)[0]  # (n_games,)
    # Build matrix: same scores for every user
    n_users = len(user_ids)
    pop_matrix = np.tile(pop_scores_1d, (n_users, 1))  # (n_users, n_games)

    results = {}
    for k in EVAL_K_VALUES:
        hr = hit_rate_at_k(pop_matrix, train_matrix, held_out, user_ids, game_ids, k)
        results[f"HR@{k}"] = round(hr, 4)
        print(f"  HR@{k} = {hr:.4f}")
    return results


# ---------------------------------------------------------------------------
# Baseline 2 — Content-based
# ---------------------------------------------------------------------------

def normalize_multilabel(value) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"[\"'\[\]{}:0-9]", " ", str(value))
    return re.sub(r"\s+", " ", text).strip()


def build_content_matrix(
    catalog: pd.DataFrame, game_ids: list[str]
) -> np.ndarray:
    """
    Build a (n_games x CONTENT_SVD_DIM) dense matrix from TF-IDF on
    name + short_description + tags + genres, reduced with TruncatedSVD.
    Aligned to game_ids order.
    """
    game_sub = catalog[catalog["AppID"].isin(game_ids)].set_index("AppID")
    texts = []
    for gid in game_ids:
        if gid in game_sub.index:
            row = game_sub.loc[gid]
            blob = (
                str(row.get("name", "")) + " "
                + str(row.get("short_description", "")) + " "
                + normalize_multilabel(row.get("tags", "")) + " "
                + normalize_multilabel(row.get("genres", "")) + " "
                + normalize_multilabel(row.get("categories", ""))
            )
            texts.append(blob)
        else:
            texts.append("")

    tfidf = TfidfVectorizer(max_features=3000, min_df=1, max_df=0.95, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(texts)

    n_comp = min(CONTENT_SVD_DIM, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    X_svd = svd.fit_transform(X_tfidf)
    X_norm = normalize(X_svd, norm="l2")
    print(f"Content matrix: {X_norm.shape}  (SVD {n_comp}D, L2-normalized)")
    return X_norm


def content_scores_for_user(
    user_train_games: list[str],
    game_ids: list[str],
    content_matrix: np.ndarray,
) -> np.ndarray:
    """Average cosine similarity from the user's train history to all games."""
    game_idx = {g: i for i, g in enumerate(game_ids)}
    train_indices = [game_idx[g] for g in user_train_games if g in game_idx]
    if not train_indices:
        return np.zeros(len(game_ids))
    user_vec = content_matrix[train_indices].mean(axis=0, keepdims=True)
    sims = cosine_similarity(user_vec, content_matrix)[0]
    return sims


def eval_content_based(
    catalog: pd.DataFrame,
    train_dict: dict[str, list[str]],
    train_matrix: sparse.csr_matrix,
    held_out: dict[str, str],
    user_ids: list[str],
    game_ids: list[str],
) -> tuple[dict, np.ndarray]:
    print("\n--- Baseline 2: Content-based ---")
    content_matrix = build_content_matrix(catalog, game_ids)

    n_users = len(user_ids)
    n_games = len(game_ids)
    scores_matrix = np.zeros((n_users, n_games), dtype=np.float32)
    user_idx = {u: i for i, u in enumerate(user_ids)}

    for uid in held_out:
        ui = user_idx[uid]
        train_games = train_dict.get(uid, [])
        scores_matrix[ui] = content_scores_for_user(train_games, game_ids, content_matrix)

    results = {}
    for k in EVAL_K_VALUES:
        hr = hit_rate_at_k(scores_matrix, train_matrix, held_out, user_ids, game_ids, k)
        results[f"HR@{k}"] = round(hr, 4)
        print(f"  HR@{k} = {hr:.4f}")
    return results, content_matrix


# ---------------------------------------------------------------------------
# System 3 — Matrix Factorization (SGD)
# ---------------------------------------------------------------------------

def train_mf(
    R_train: np.ndarray,
    k: int,
    rng: np.random.Generator,
    lr: float = MF_LR,
    reg: float = MF_REG,
    epochs: int = MF_EPOCHS,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    SGD matrix factorization on binary interaction matrix R_train.

    Objective:
        min_{P,Q} sum_{(u,i) in Omega} (R_ui - p_u^T q_i)^2
                  + lambda * (||P||_F^2 + ||Q||_F^2)

    Update rules (per observed pair):
        p_u <- p_u + lr * (e_ui * q_i - reg * p_u)
        q_i <- q_i + lr * (e_ui * p_u - reg * q_i)
    """
    n_users, n_items = R_train.shape
    P = 0.1 * rng.standard_normal((n_users, k)).astype(np.float32)
    Q = 0.1 * rng.standard_normal((n_items, k)).astype(np.float32)

    observed = np.argwhere(R_train > 0)
    loss_history = []

    for epoch in range(epochs):
        rng.shuffle(observed)
        for u, i in observed:
            pred = float(P[u] @ Q[i])
            err = float(R_train[u, i]) - pred
            p_old = P[u].copy()
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * p_old - reg * Q[i])

        pred_mat = P @ Q.T
        obs_pred = pred_mat[R_train > 0]
        obs_true = R_train[R_train > 0]
        loss = float(np.mean((obs_true - obs_pred) ** 2)) + reg * (
            float(np.mean(P ** 2)) + float(np.mean(Q ** 2))
        )
        loss_history.append(loss)

    return P, Q, loss_history


def eval_mf(
    R_train_full: sparse.csr_matrix,
    held_out: dict[str, str],
    user_ids: list[str],
    game_ids: list[str],
    k_dim: int,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray, np.ndarray, list[float]]:
    """Train and evaluate MF for a given latent dimension k_dim."""
    # Use dense sub-matrix for qualifying users only to keep memory tractable
    qualifying_users = list(held_out.keys())
    user_idx_all = {u: i for i, u in enumerate(user_ids)}
    q_indices = [user_idx_all[u] for u in qualifying_users]

    # Dense sub-matrix: qualifying users x all games
    R_sub = R_train_full[q_indices].toarray().astype(np.float32)

    sub_rng = np.random.default_rng(RANDOM_STATE + k_dim)
    P, Q, loss_history = train_mf(R_sub, k=k_dim, rng=sub_rng)

    # Build score matrix for qualifying users
    pred_matrix = P @ Q.T  # (n_qualifying x n_games)

    # Rebuild sparse sub-matrix for masking
    R_sub_sparse = sparse.csr_matrix(R_sub)

    # Map back to full user indices for hit_rate_at_k
    # We need a (n_users x n_games) scores matrix, but only qualifying rows matter
    n_users = len(user_ids)
    n_games = len(game_ids)
    scores_full = np.zeros((n_users, n_games), dtype=np.float32)
    for sub_i, global_i in enumerate(q_indices):
        scores_full[global_i] = pred_matrix[sub_i]

    results = {}
    for k_eval in EVAL_K_VALUES:
        hr = hit_rate_at_k(scores_full, R_train_full, held_out, user_ids, game_ids, k_eval)
        results[f"HR@{k_eval}"] = round(hr, 4)

    print(f"  k={k_dim:>2}  " + "  ".join(f"HR@{k}={results[f'HR@{k}']:.4f}" for k in EVAL_K_VALUES)
          + f"  loss_final={loss_history[-1]:.4f}")

    return results, P, Q, loss_history


def run_mf_sweep(
    train_matrix: sparse.csr_matrix,
    held_out: dict[str, str],
    user_ids: list[str],
    game_ids: list[str],
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray, np.ndarray, list[float], int]:
    print("\n--- System 3: Matrix Factorization (SGD) ---")
    print(f"  epochs={MF_EPOCHS}  lr={MF_LR}  reg={MF_REG}")
    all_results = {}
    best_hr10 = -1.0
    best_P, best_Q, best_loss, best_k = None, None, [], 0

    for k_dim in MF_K_VALUES:
        results, P, Q, loss_history = eval_mf(
            train_matrix, held_out, user_ids, game_ids, k_dim, rng
        )
        all_results[f"k={k_dim}"] = results
        hr10 = results["HR@10"]
        if hr10 > best_hr10:
            best_hr10 = hr10
            best_P, best_Q, best_loss, best_k = P, Q, loss_history, k_dim

    return all_results, best_P, best_Q, best_loss, best_k


# ---------------------------------------------------------------------------
# System 4 — Hybrid (content + MF)
# ---------------------------------------------------------------------------

def eval_hybrid(
    best_P: np.ndarray,
    best_Q: np.ndarray,
    content_matrix: np.ndarray,
    train_dict: dict[str, list[str]],
    train_matrix: sparse.csr_matrix,
    held_out: dict[str, str],
    user_ids: list[str],
    game_ids: list[str],
) -> dict:
    """
    Hybrid score: alpha * content_score + (1 - alpha) * mf_score.

    Data alignment: AppID is the common key for both systems. The content
    matrix and MF item factors Q are both indexed by game_ids order.
    No cross-catalog mapping is needed — alignment is exact.
    """
    print("\n--- System 4: Hybrid (content + MF, alpha sweep) ---")
    user_idx = {u: i for i, u in enumerate(user_ids)}
    qualifying_users = list(held_out.keys())
    q_indices = [user_idx[u] for u in qualifying_users]

    # MF scores for qualifying users
    mf_scores = best_P @ best_Q.T  # (n_qualifying x n_games)
    mf_min, mf_max = mf_scores.min(), mf_scores.max()
    mf_norm = (mf_scores - mf_min) / (mf_max - mf_min + 1e-9)

    # Content scores for qualifying users
    n_games = len(game_ids)
    content_scores = np.zeros((len(qualifying_users), n_games), dtype=np.float32)
    for sub_i, uid in enumerate(qualifying_users):
        train_games = train_dict.get(uid, [])
        content_scores[sub_i] = content_scores_for_user(train_games, game_ids, content_matrix)

    c_min, c_max = content_scores.min(), content_scores.max()
    content_norm = (content_scores - c_min) / (c_max - c_min + 1e-9)

    all_results = {}
    for alpha in HYBRID_ALPHAS:
        hybrid = alpha * content_norm + (1 - alpha) * mf_norm  # (n_qualifying x n_games)

        # Build full scores matrix
        n_users = len(user_ids)
        scores_full = np.zeros((n_users, n_games), dtype=np.float32)
        for sub_i, global_i in enumerate(q_indices):
            scores_full[global_i] = hybrid[sub_i]

        results = {}
        for k_eval in EVAL_K_VALUES:
            hr = hit_rate_at_k(scores_full, train_matrix, held_out, user_ids, game_ids, k_eval)
            results[f"HR@{k_eval}"] = round(hr, 4)

        all_results[f"alpha={alpha}"] = results
        print(f"  alpha={alpha:.1f}  " + "  ".join(
            f"HR@{k}={results[f'HR@{k}']:.4f}" for k in EVAL_K_VALUES
        ))

    return all_results


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(
    best_P: np.ndarray,
    best_Q: np.ndarray,
    catalog: pd.DataFrame,
    train_dict: dict[str, list[str]],
    held_out: dict[str, str],
    user_ids: list[str],
    game_ids: list[str],
    train_matrix: sparse.csr_matrix,
    n_examples: int = 5,
) -> dict:
    """Show concrete hit and miss examples for the best MF model."""
    user_idx = {u: i for i, u in enumerate(user_ids)}
    game_idx = {g: i for i, g in enumerate(game_ids)}
    game_names = catalog.set_index("AppID")["name"].to_dict()

    pred_full = np.zeros((len(user_ids), len(game_ids)), dtype=np.float32)
    qualifying = list(held_out.keys())
    q_indices = [user_idx[u] for u in qualifying]
    pred_sub = best_P @ best_Q.T
    for si, gi in enumerate(q_indices):
        pred_full[gi] = pred_sub[si]

    hits_examples = []
    miss_examples = []

    for uid, held_game in held_out.items():
        if held_game not in game_idx:
            continue
        ui = user_idx[uid]
        scores = pred_full[ui].copy()
        scores[train_matrix[ui].indices] = -np.inf
        top10 = list(np.argsort(scores)[::-1][:10])
        held_idx = game_idx[held_game]
        is_hit = held_idx in top10

        example = {
            "user": uid,
            "held_out_game": game_names.get(held_game, held_game),
            "train_games": [game_names.get(g, g) for g in train_dict.get(uid, [])[:3]],
            "top3_recommended": [game_names.get(game_ids[i], game_ids[i]) for i in top10[:3]],
            "hit": is_hit,
        }

        if is_hit and len(hits_examples) < n_examples:
            hits_examples.append(example)
        elif not is_hit and len(miss_examples) < n_examples:
            miss_examples.append(example)

        if len(hits_examples) >= n_examples and len(miss_examples) >= n_examples:
            break

    return {"hit_examples": hits_examples, "miss_examples": miss_examples}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    # Load data
    interactions, catalog = load_interactions()
    pos_df, user_ids, game_ids = build_interaction_universe(interactions, catalog)

    # Train/test split
    train_dict, held_out = train_test_split(pos_df, user_ids, rng)
    print(f"\nHeld-out users: {len(held_out):,}")

    # Sparse train matrix
    train_matrix = build_sparse_matrix(train_dict, user_ids, game_ids)

    # Evaluate baselines
    pop_results = eval_popularity(catalog, train_matrix, held_out, user_ids, game_ids)
    content_results, content_matrix = eval_content_based(
        catalog, train_dict, train_matrix, held_out, user_ids, game_ids
    )

    # Matrix Factorization sweep
    mf_all_results, best_P, best_Q, best_loss, best_k = run_mf_sweep(
        train_matrix, held_out, user_ids, game_ids, rng
    )

    # Hybrid sweep
    hybrid_results = eval_hybrid(
        best_P, best_Q, content_matrix, train_dict,
        train_matrix, held_out, user_ids, game_ids,
    )

    # Error analysis
    print("\n--- Error analysis (MF best model) ---")
    errors = error_analysis(
        best_P, best_Q, catalog, train_dict, held_out,
        user_ids, game_ids, train_matrix,
    )
    print(f"  Hit examples: {len(errors['hit_examples'])}")
    print(f"  Miss examples: {len(errors['miss_examples'])}")

    # Summary comparison
    print("\n=== Summary: HR@10 comparison ===")
    print(f"  Popularity baseline : {pop_results['HR@10']:.4f}")
    print(f"  Content-based       : {content_results['HR@10']:.4f}")
    for cfg, res in mf_all_results.items():
        print(f"  MF ({cfg:<6})        : {res['HR@10']:.4f}")
    best_hybrid = max(hybrid_results.items(), key=lambda x: x[1]["HR@10"])
    print(f"  Hybrid best ({best_hybrid[0]}): {best_hybrid[1]['HR@10']:.4f}")

    # Save artifacts
    np.save(os.path.join(ARTIFACTS_DIR, "mf_P.npy"), best_P)
    np.save(os.path.join(ARTIFACTS_DIR, "mf_Q.npy"), best_Q)
    np.save(os.path.join(ARTIFACTS_DIR, "user_index.npy"), np.array(user_ids))
    np.save(os.path.join(ARTIFACTS_DIR, "item_index.npy"), np.array(game_ids))
    np.save(os.path.join(ARTIFACTS_DIR, "content_matrix.npy"), content_matrix)

    metrics = {
        "n_users": len(user_ids),
        "n_games": len(game_ids),
        "n_qualifying_users": len(held_out),
        "interaction_matrix_shape": [len(user_ids), len(game_ids)],
        "mf_best_k": best_k,
        "mf_loss_history": best_loss,
        "evaluation": {
            "popularity_baseline": pop_results,
            "content_based_baseline": content_results,
            "matrix_factorization": mf_all_results,
            "hybrid": hybrid_results,
        },
        "error_analysis": errors,
    }
    with open(os.path.join(ARTIFACTS_DIR, "rec_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    run_pipeline()
