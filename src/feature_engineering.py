"""
Week 5 - Reproducible Feature Engineering & Dimensionality Reduction Pipeline.

Builds numeric, categorical, text, and temporal feature matrices from the Steam
catalog and review aggregates, fits PCA on numeric features and TruncatedSVD on
TF-IDF text features, computes reconstruction errors, and persists matrices,
fitted models and metrics under ``artifacts/``.

Run from project root:

    python src/feature_engineering.py
"""

from __future__ import annotations

import json
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "catalog", "games_may2024_full.csv")
PARQUET_FULL = os.path.join(PROJECT_ROOT, "data", "processed", "steam_full.parquet")
PARQUET_V1 = os.path.join(PROJECT_ROOT, "data", "processed", "steam_v1.parquet")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

NUMERIC_BASE_COLS = [
    "required_age", "price", "dlc_count", "metacritic_score",
    "achievements", "recommendations", "user_score", "score_rank",
    "positive", "negative", "estimated_owners", "average_playtime_forever",
    "average_playtime_2weeks", "median_playtime_forever", "median_playtime_2weeks",
    "peak_ccu", "pct_pos_total", "num_reviews_total", "pct_pos_recent",
    "num_reviews_recent",
]
PLATFORM_COLS = ["windows", "mac", "linux"]
CATEGORICAL_COLS = [
    "genres", "categories", "tags", "developers", "publishers",
    "supported_languages", "full_audio_languages",
]
TEXT_COLS = ["name", "short_description", "about_the_game", "detailed_description"]
REVIEW_AGG_COLS = ["reviews_count", "voted_up_rate", "playtime_mean", "playtime_median"]


@dataclass
class FeatureMatrices:
    X_numeric: np.ndarray
    X_categorical: sparse.csr_matrix
    X_text: sparse.csr_matrix
    numeric_columns: list[str] = field(default_factory=list)
    categorical_classes: list[str] = field(default_factory=list)
    text_vocabulary: list[str] = field(default_factory=list)


def normalize_multilabel(value) -> list[str]:
    """Parse stringified lists/dicts into a clean token list."""
    if pd.isna(value):
        return []
    text = str(value).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    parts = re.split(r"[;,]", text)
    return [p.strip() for p in parts if p.strip()]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(
            f"Catalog CSV not found at {CATALOG_PATH}. Run ingestion first."
        )

    catalog_df = pd.read_csv(CATALOG_PATH)
    if os.path.exists(PARQUET_FULL):
        parquet_path = PARQUET_FULL
    elif os.path.exists(PARQUET_V1):
        parquet_path = PARQUET_V1
    else:
        raise FileNotFoundError(
            f"No parquet dataset found at {PARQUET_FULL} or {PARQUET_V1}. "
            "Run ingestion first."
        )

    reviews_df = pd.read_parquet(parquet_path)
    catalog_df["AppID"] = catalog_df["AppID"].astype(str)
    reviews_df["AppID"] = reviews_df["AppID"].astype(str)
    return catalog_df, reviews_df, parquet_path


def merge_catalog_with_review_aggregates(
    catalog_df: pd.DataFrame, reviews_df: pd.DataFrame
) -> pd.DataFrame:
    reviews = reviews_df[["AppID", "voted_up", "author_playtime_forever"]].copy()
    reviews["voted_up"] = reviews["voted_up"].astype(float)
    agg = reviews.groupby("AppID", as_index=False).agg(
        reviews_count=("voted_up", "size"),
        voted_up_rate=("voted_up", "mean"),
        playtime_mean=("author_playtime_forever", "mean"),
        playtime_median=("author_playtime_forever", "median"),
    )
    return catalog_df.merge(agg, on="AppID", how="left")


def build_numeric_features(
    df: pd.DataFrame, include_review_aggregates: bool = False
) -> tuple[np.ndarray, list[str], StandardScaler]:
    """
    Numeric + temporal features, standardized.

    By design we exclude review-aggregate columns when the V1 parquet only
    covers a stratified subset of games — they would be NaN for ~99% of the
    catalog and add no signal. Set ``include_review_aggregates=True`` once the
    full parquet is available.
    """
    work = df.copy()
    cols = list(NUMERIC_BASE_COLS)

    for c in PLATFORM_COLS:
        work[c] = pd.to_numeric(work[c], errors="coerce").astype(float)
        cols.append(c)

    work["release_date"] = pd.to_datetime(work["release_date"], errors="coerce", utc=True).dt.tz_localize(None)
    work["release_year"] = work["release_date"].dt.year
    work["release_month"] = work["release_date"].dt.month
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    work["release_age_days"] = (today - work["release_date"]).dt.days
    cols += ["release_year", "release_month", "release_age_days"]

    if include_review_aggregates:
        cols += REVIEW_AGG_COLS

    numeric_df = work[cols].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True)).fillna(0)

    # Outlier clipping at 1/99 percentiles stabilizes PCA against extreme tails
    # (a few AAA outliers otherwise dominate the principal directions).
    lower = numeric_df.quantile(0.01)
    upper = numeric_df.quantile(0.99)
    numeric_df = numeric_df.clip(lower, upper, axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(numeric_df.values)
    return X, cols, scaler


def build_categorical_features(
    df: pd.DataFrame, min_freq: int = 20, max_features: int = 5000
) -> tuple[sparse.csr_matrix, list[str], MultiLabelBinarizer]:
    rows: list[list[str]] = []
    counts: Counter = Counter()
    for _, row in df[CATEGORICAL_COLS].iterrows():
        tokens = []
        for col in CATEGORICAL_COLS:
            for val in normalize_multilabel(row[col]):
                tokens.append(f"{col}__{val.lower()}")
        rows.append(tokens)
        counts.update(tokens)

    allowed = [t for t, c in counts.items() if c >= min_freq]
    if max_features is not None and len(allowed) > max_features:
        allowed = [t for t, _ in counts.most_common(max_features)]
    allowed_set = set(allowed)
    rows = [[t for t in r if t in allowed_set] for r in rows]

    mlb = MultiLabelBinarizer(classes=sorted(allowed_set), sparse_output=True)
    X = mlb.fit_transform(rows)
    return X.tocsr(), list(mlb.classes_), mlb


def build_text_features(
    df: pd.DataFrame, max_features: int = 5000
) -> tuple[sparse.csr_matrix, list[str], TfidfVectorizer]:
    text = df[TEXT_COLS].fillna("")
    blob = (
        text["name"] + " " + text["short_description"] + " "
        + text["about_the_game"] + " " + text["detailed_description"]
    )
    tfidf = TfidfVectorizer(
        max_features=max_features, min_df=3, max_df=0.95, ngram_range=(1, 2)
    )
    X = tfidf.fit_transform(blob.values)
    return X, list(tfidf.get_feature_names_out()), tfidf


def fit_pca(X: np.ndarray, n_components: int = 30) -> tuple[PCA, dict]:
    n_components = min(n_components, X.shape[1], X.shape[0])
    model = PCA(n_components=n_components, random_state=42)
    Xp = model.fit_transform(X)
    Xrec = model.inverse_transform(Xp)
    frob_full = float(np.linalg.norm(X, ord="fro"))
    frob_err = float(np.linalg.norm(X - Xrec, ord="fro"))
    metrics = {
        "n_components": int(n_components),
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": np.cumsum(model.explained_variance_ratio_).tolist(),
        "components_for_80pct": int(np.searchsorted(np.cumsum(model.explained_variance_ratio_), 0.80) + 1),
        "components_for_90pct": int(np.searchsorted(np.cumsum(model.explained_variance_ratio_), 0.90) + 1),
        "components_for_95pct": int(np.searchsorted(np.cumsum(model.explained_variance_ratio_), 0.95) + 1),
        "reconstruction_frobenius": frob_err,
        "relative_reconstruction_error": frob_err / frob_full if frob_full > 0 else None,
    }
    return model, metrics


def fit_svd(X: sparse.csr_matrix, n_components: int = 200) -> tuple[TruncatedSVD, dict]:
    max_components = min(X.shape[1] - 1, X.shape[0] - 1)
    if max_components < 1:
        raise ValueError(
            "TruncatedSVD requires at least 2 samples and 2 features in the input matrix."
        )
    n_components = min(n_components, max_components)
    model = TruncatedSVD(n_components=n_components, random_state=42)
    Xp = model.fit_transform(X)
    # Reconstruction in the original sparse space using V^T
    Xrec = Xp @ model.components_
    diff = X - Xrec
    frob_full = float(sparse.linalg.norm(X, ord="fro"))
    frob_err = float(np.linalg.norm(diff, ord="fro"))
    metrics = {
        "n_components": int(n_components),
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": np.cumsum(model.explained_variance_ratio_).tolist(),
        "components_for_50pct": int(np.searchsorted(np.cumsum(model.explained_variance_ratio_), 0.50) + 1),
        "components_for_80pct": int(np.searchsorted(np.cumsum(model.explained_variance_ratio_), 0.80) + 1),
        "components_for_90pct": int(np.searchsorted(np.cumsum(model.explained_variance_ratio_), 0.90) + 1),
        "reconstruction_frobenius": frob_err,
        "relative_reconstruction_error": frob_err / frob_full if frob_full > 0 else None,
    }
    return model, metrics


def save_artifacts(
    matrices: FeatureMatrices,
    pca_model: PCA,
    svd_model: TruncatedSVD,
    scaler: StandardScaler,
    mlb: MultiLabelBinarizer,
    tfidf: TfidfVectorizer,
    metrics: dict,
) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    np.savez_compressed(
        os.path.join(ARTIFACTS_DIR, "X_numeric.npz"),
        X=matrices.X_numeric,
        columns=np.array(matrices.numeric_columns, dtype=object),
    )
    sparse.save_npz(os.path.join(ARTIFACTS_DIR, "X_categorical.npz"), matrices.X_categorical)
    sparse.save_npz(os.path.join(ARTIFACTS_DIR, "X_text.npz"), matrices.X_text)

    with open(os.path.join(ARTIFACTS_DIR, "models.pkl"), "wb") as f:
        pickle.dump(
            {
                "scaler": scaler, "mlb": mlb, "tfidf": tfidf,
                "pca": pca_model, "svd": svd_model,
            },
            f,
        )

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def run_pipeline(include_review_aggregates: bool = False) -> dict:
    print("Loading data...")
    catalog_df, reviews_df, parquet_path = load_data()
    df = merge_catalog_with_review_aggregates(catalog_df, reviews_df)
    print(f"Catalog rows: {len(catalog_df)} | Reviews source: {parquet_path}")

    print("Building numeric features...")
    X_num, num_cols, scaler = build_numeric_features(
        df, include_review_aggregates=include_review_aggregates
    )
    print(f"  X_numeric: {X_num.shape}")

    print("Building categorical features...")
    X_cat, cat_classes, mlb = build_categorical_features(df)
    print(f"  X_categorical: {X_cat.shape}")

    print("Building text features...")
    X_text, vocab, tfidf = build_text_features(df)
    print(f"  X_text: {X_text.shape}")

    matrices = FeatureMatrices(
        X_numeric=X_num, X_categorical=X_cat, X_text=X_text,
        numeric_columns=num_cols, categorical_classes=cat_classes, text_vocabulary=vocab,
    )

    print("Fitting PCA on numeric features...")
    pca_model, pca_metrics = fit_pca(X_num, n_components=min(X_num.shape[1], 30))
    print(f"  k for 90% variance: {pca_metrics['components_for_90pct']}")
    print(f"  Relative reconstruction error: {pca_metrics['relative_reconstruction_error']:.4f}")

    print("Fitting TruncatedSVD on TF-IDF text features...")
    svd_model, svd_metrics = fit_svd(X_text, n_components=200)
    print(f"  k for 50% variance: {svd_metrics['components_for_50pct']}")
    print(f"  Relative reconstruction error: {svd_metrics['relative_reconstruction_error']:.4f}")

    metrics = {
        "shapes": {
            "numeric": list(X_num.shape),
            "categorical": list(X_cat.shape),
            "text": list(X_text.shape),
        },
        "include_review_aggregates": include_review_aggregates,
        "parquet_used": parquet_path,
        "pca_numeric": pca_metrics,
        "svd_text": svd_metrics,
    }

    print("Saving artifacts...")
    save_artifacts(matrices, pca_model, svd_model, scaler, mlb, tfidf, metrics)
    print(f"Artifacts written to: {ARTIFACTS_DIR}")
    return metrics


if __name__ == "__main__":
    run_pipeline()
