# Semester Project: Domain Discovery, Recommendation, and Graph Intelligence
## Milestone: Representation and Dimensionality Report (Week 5)

---

### Team 6 Members

| # | Team Member |
| --- | --- |
| 1 | Iam Anthony Marcelo Alvarez Orellana |
| 2 | Jeffrey Ulises Diaz Villanueva |
| 3 | Paula Jimena Mancilla Cienfuegos |
| 4 | Fernando Samuel Paredes Espinoza |

---

### 1. Feature Matrices

Three complementary feature blocks were built from `data/raw/catalog/games_may2024_full.csv` (~88,000 games). Each block captures a distinct dimension of a videogame and is stored under `artifacts/`.

| Block | Artifact | Shape | Type | Construction |
| :--- | :--- | :--- | :--- | :--- |
| Numeric + temporal | `X_numeric.npz` | 88k × 23 | Dense `float64` | StandardScaler + 1/99 percentile clipping |
| Categorical (tags/genres) | `X_categorical.npz` | 88k × 5,000 | Sparse CSR | MultiLabelBinarizer, `min_freq=20` |
| Text (descriptions) | `X_text.npz` | 88k × 5,000 | Sparse CSR | TfidfVectorizer, 1–2 grams, `min_df=3`, `max_df=0.95` |

#### 1.1 Numeric and Temporal Features (23 columns)

Numeric features capture scale-sensitive signals: price, popularity, engagement, and platform availability. Temporal features (release year, release month, age in days) encode lifecycle effects. All values are standardized with `StandardScaler` after clipping extreme values at the 1st and 99th percentiles.

Clipping is necessary because a small number of AAA titles have playtime, review counts, and revenue estimates that are orders of magnitude above the median indie game. Without clipping, those outliers dominate the first principal component and obscure the structure of the remaining 87,000+ games.

Numeric columns included:
`required_age`, `price`, `dlc_count`, `metacritic_score`, `achievements`, `recommendations`, `user_score`, `score_rank`, `positive`, `negative`, `estimated_owners`, `average_playtime_forever`, `average_playtime_2weeks`, `median_playtime_forever`, `median_playtime_2weeks`, `peak_ccu`, `pct_pos_total`, `num_reviews_total`, `pct_pos_recent`, `num_reviews_recent`, `windows`, `mac`, `linux`.

#### 1.2 Categorical Features (multi-label binary, up to 5,000 columns)

Fields `genres`, `categories`, `tags`, `developers`, `publishers`, `supported_languages`, and `full_audio_languages` are tokenized using a custom `normalize_multilabel` parser that handles stringified Python lists and dicts. Each token is prefixed by field name (e.g., `tags__roguelike`, `genres__action`) to avoid cross-field collisions. Tokens appearing in fewer than 20 games are filtered out to reduce noise.

#### 1.3 Text Features (TF-IDF, up to 5,000 terms)

Game name, short description, about-the-game, and detailed description are concatenated and vectorized with `TfidfVectorizer`. Unigrams and bigrams are included. Terms appearing in fewer than 3 documents or more than 95% of documents are excluded. TF-IDF emphasizes terms that are informative about a specific game while downweighting generic words that appear across the catalog.

---

### 2. Dimensionality Reduction

#### 2.1 PCA on X_numeric

Principal Component Analysis was applied to the dense numerical block. The scikit-learn `PCA` implementation performs full SVD internally, so the result is exact (not approximate).

Key findings:
- **10 components** preserve approximately **90% of total variance**.
- The first component is dominated by popularity and engagement signals (`positive`, `recommendations`, `num_reviews_total`), which are highly correlated.
- The Frobenius relative reconstruction error with 10 components is approximately **0.10**, meaning 10% of the total Frobenius norm is lost — an acceptable trade-off for compression from 23 to 10 dimensions.

Clipping at the 1/99 percentiles was the single preprocessing choice that most affected PCA: without it, PC1 explained >50% of variance and was entirely driven by the handful of AAA outliers, making the remaining components nearly uninterpretable.

Fitted model saved to `artifacts/models.pkl` under key `pca`.

#### 2.2 TruncatedSVD on X_categorical

For the sparse tag/genre matrix, standard PCA is unsuitable: centering a sparse matrix would densify it and exhaust memory at 88k × 5,000 scale. `TruncatedSVD` operates directly on the sparse CSR representation via randomized SVD.

After filtering tags with frequency below 30, **493 unique tag dimensions** survived. These were reduced to **50 latent components**, retaining approximately **62% of cumulative explained variance** — a favorable trade-off between fidelity and compactness for downstream clustering.

The 50-component representation is used as input to the Week 7 clustering pipeline.

#### 2.3 TruncatedSVD on X_text

The text feature matrix (88k × 5,000 sparse TF-IDF) was reduced to **200 latent semantic components** using `TruncatedSVD`. This produces a dense semantic embedding per game that captures co-occurrence structure between terms across descriptions.

Fitted models saved to `artifacts/models.pkl` under keys `svd` (text SVD) and `mlb`, `tfidf` (transformers).

---

### 3. Comparison Table

| Representation | Input shape | Reduction method | Output dim | Variance retained | Relative reconstruction error |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Numeric | 88k × 23 | PCA | 10 | ~90% | ~0.10 |
| Categorical (tags) | 88k × 493* | TruncatedSVD | 50 | ~62% | — |
| Text (TF-IDF) | 88k × 5,000 | TruncatedSVD | 200 | — | — |

*After filtering tokens with frequency < 30.

**Interpretation:** The numeric block is the most compressible: 10 dimensions suffice for 90% of variance because many columns are correlated proxies for the same underlying signal (popularity, engagement, reception). The categorical tag block is less compressible — 50 dimensions capture only 62% of variance — reflecting that the Steam tag taxonomy has genuine diversity: a game's identity in terms of genre, mechanics, and mood is not easily collapsed into a handful of latent themes. The text block requires 200 dimensions because free-text descriptions are high-dimensional and sparse; reconstruction error was not the primary objective here, and the 200-component representation is used downstream for content-based similarity.

---

### 4. Visualizations

The following plots are generated by running `python src/feature_engineering.py` or executing `src/feature_engineering_week5.ipynb`. Figures are displayed inline in the notebook.

1. **PCA Explained Variance Ratio** — bar chart of individual variance per component, showing the steep drop-off after the first 3–4 components.
2. **PCA Cumulative Explained Variance** — line chart showing the elbow at ~10 components reaching 90%.
3. **PCA Scatter (PC1 vs PC2, colored by voted_up_rate)** — 2D projection of all 88k games, colored by aggregated recommendation rate from the interaction layer. Reveals that high-rated and low-rated games do not form cleanly separable clusters in commercial-numeric space alone, motivating tag-based representations for clustering.
4. **SVD Explained Variance Ratio (Text)** — bar chart showing slow decay in semantic text components, consistent with the diversity of game descriptions.
5. **SVD Cumulative Explained Variance (Text)** — line chart; 200 components do not reach 80%, confirming high intrinsic dimensionality of free-text game descriptions.

---

### 5. Technical Interpretation

**What was learned about dimensionality:**

The numeric block is low-rank by nature: most numeric signals are proxies for commercial popularity. PCA exposes this redundancy efficiently, and the 10-component compression is well-justified. Retaining more components adds noise rather than signal in the context of game discovery.

The categorical tag block has medium intrinsic dimensionality: 50 components capture 62% of variance but the remaining 38% is not pure noise — it contains rare genre combinations (e.g., "local co-op puzzle platformer") that matter for niche discovery. This motivates using a larger latent space (50D rather than 10D) before clustering.

The text block has high intrinsic dimensionality. TF-IDF descriptions of games do not share vocabulary in a way that permits aggressive compression. This is expected: unlike news articles about recurring events, game descriptions span unique mechanics, narrative universes, and interaction paradigms.

**What the three representations preserve:**

- `X_numeric` preserves commercial and engagement structure. Useful for popularity-aware ranking.
- `X_categorical` (SVD 50D) preserves semantic type structure. Useful for similarity and clustering by game identity.
- `X_text` (SVD 200D) preserves lexical-semantic structure. Useful for content-based recommendation and cold-start scenarios.

**Representation limits:**

- PCA on numeric data does not capture non-linear structure. A game with unusual pricing or engagement patterns may project onto components that are not semantically meaningful.
- TF-IDF ignores word order and context. Two games can share vocabulary but have opposite tones (e.g., "horror" as setting vs. "horror" as a negative review term).
- Neither representation captures user behavioral signals: those are encoded in the interaction layer (`steam_v1.parquet`) and are reserved for the Week 10 collaborative filtering pipeline.

---

### 6. Reproducible Pipeline

The full feature-building pipeline runs from a single command at the project root:

```bash
# 1. Activate virtual environment
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 2. Run feature engineering pipeline
python src/feature_engineering.py
```

Outputs written to `artifacts/`:
- `X_numeric.npz` — dense numeric feature matrix
- `X_categorical.npz` — sparse categorical feature matrix
- `X_text.npz` — sparse TF-IDF text feature matrix
- `models.pkl` — fitted `StandardScaler`, `MultiLabelBinarizer`, `TfidfVectorizer`, `PCA`, `TruncatedSVD`
- `metrics.json` — explained variance ratios, reconstruction errors, component thresholds

The notebook `src/feature_engineering_week5.ipynb` reproduces the same pipeline interactively and generates all visualizations inline.

---

### 7. Ethics and Access Note

* **Data provenance:** All features are derived from `games_may2024_full.csv`, sourced from the public Steamworks API via Kaggle (CC0 license, dataset by Artemiy Ermilov).
* **Authorization:** CC0 license permits unrestricted academic use and redistribution.
* **Personal data risks:** This milestone operates exclusively on game catalog metadata. No user identifiers (`author_steamid`) are present in the feature matrices. Personal data risk at this stage is negligible.
* **Risk mitigation:** The `author_steamid` field, present in the interaction parquet, is not included in any feature matrix built here. It is reserved for the interaction layer and treated as an opaque identifier with no linkage to real-world identities.
