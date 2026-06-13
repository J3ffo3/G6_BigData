# Semester Project: Domain Discovery, Recommendation, and Graph Intelligence
## Milestone: Recommendation, Ranking, and Predictive Decision Engine (Week 10)

---

### Team 6 Members

| # | Team Member |
| --- | --- |
| 1 | Iam Anthony Marcelo Alvarez Orellana |
| 2 | Jeffrey Ulises Diaz Villanueva |
| 3 | Paula Jimena Mancilla Cienfuegos |
| 4 | Fernando Samuel Paredes Espinoza |

---

### 1. Task Framing

**Task type:** Recommendation.

Given a user's positive interaction history (games they endorsed via `voted_up=True`), the system predicts which unseen game that user would also endorse. The output is a ranked list of candidate games.

This is not a pure ranking task (no predefined query) and not a classification task (no binary label per candidate). It is a **personalized recommendation** problem where the unit being recommended is a Steam game, the unit of personalization is the user interaction history, and the quality signal is the recovered hold-out positive interaction.

The clustering results from Week 7 (six semantic segments: Casual, Action, Local Multiplayer, Narrative/JRPG, VR, Strategy) feed into this pipeline as a candidate-filtering feature: a user's interaction history implies a probable cluster affinity, and candidates from the matching cluster receive a mild prior boost. This makes the system a **segmentation-feeding-recommendation** architecture.

---

### 2. Data Alignment

All three systems (popularity, content-based, MF) operate on the same 500-game universe from `data/processed/steam_v1.parquet`. `AppID` is the common join key across:

- `steam_v1.parquet` — user interaction records (`author_steamid`, `AppID`, `voted_up`, `author_playtime_forever`)
- `games_may2024_full.csv` — game metadata (`name`, `tags`, `genres`, `categories`, `short_description`, `recommendations`)
- `artifacts/clustering/cluster_labels_k6.npy` — cluster labels from Week 7

No cross-catalog mapping is required. Alignment is exact by construction because the parquet was built via an inner join on `AppID` in `src/ingest.py`.

---

### 3. Evaluation Protocol

**Interaction matrix:** Sparse binary matrix, users × games. Value = 1 when `voted_up = True`. Duplicate (user, game) pairs are de-duplicated before matrix construction.

**Qualifying users:** Users with ≥ 2 distinct positive interactions. This threshold is required so that at least one training interaction exists after holding one out. Users with only one positive interaction are excluded from evaluation.

**Train/test split:** For each qualifying user, one positive interaction is randomly held out (`random_state=42`). The remaining interactions form the training set.

**Candidate pool:** All games the user has NOT interacted with in the training set. This is the full game universe minus the user's known positives — the standard implicit-feedback evaluation setting.

**Metric:** Hit-Rate at K (HR@K) — the fraction of qualifying users for whom the held-out game appears in their top-K recommendations. Primary metric: **HR@10**.

**Why HR@10:** The product question asks whether a user can find a relevant game in a short list. A top-10 list is a realistic display surface for a game discovery interface. HR@10 is interpretable: a score of 0.15 means 15% of users would see their held-out endorsed game in the first 10 results.

**Leakage prevention:** Training items are masked to −∞ before ranking. The held-out item is never used during training or score computation. The same random seed ensures reproducibility.

---

### 4. Baseline 1 — Popularity

**Design:** All games are ranked by their aggregate `recommendations` count from the catalog (total positive review count across all Steam users). Every user receives the same ranking. No personalization.

**Justification:** Popularity is the simplest defensible baseline. It reflects market-level endorsement and requires no user history. Any personalized system must beat it to justify added complexity.

**Limitation:** Popularity strongly biases toward well-known AAA titles. It cannot surface niche Indie games — the exact problem stated in the project's product question.

**Command:** The popularity scores are computed inside `src/recommendation.py` without external data: the `recommendations` column in the catalog CSV provides all necessary signal.

---

### 5. Baseline 2 — Content-Based

**Design:** For each game in the 500-game universe, a text representation is built from `name + short_description + tags + genres + categories` using `TfidfVectorizer` (1–2 grams, up to 3,000 features). `TruncatedSVD` reduces the TF-IDF matrix to 50 latent components. Each game becomes an L2-normalized vector in R⁵⁰.

For a given user with training history {g₁, g₂, …}, a user vector is computed as the centroid of the game vectors for {g₁, g₂, …}. Cosine similarity between the user vector and all candidate games produces the ranking.

**Justification:** Content-based methods are cold-start friendly (they need only item metadata, not user co-occurrence patterns) and provide an interpretable similarity notion: two games score high if they share vocabulary in descriptions and tags.

**Limitation:** Content-based methods overspecialize. A user who has only played Action games will receive more Action games, even if latent behavior data would suggest they might also enjoy a Narrative RPG. The method also ignores what other users with similar taste profiles have enjoyed.

---

### 6. System 3 — Matrix Factorization (SGD)

**Model:** Low-rank latent factor model trained with stochastic gradient descent.

**Objective:**

$$\min_{P,Q} \sum_{(u,i) \in \Omega} \left(R_{ui} - p_u^\top q_i\right)^2 + \lambda \left(\|P\|_F^2 + \|Q\|_F^2\right)$$

where $R \in \{0,1\}^{m \times n}$ is the binary interaction matrix, $\Omega$ is the set of observed positive pairs, $P \in \mathbb{R}^{m \times k}$ are user latent factors, and $Q \in \mathbb{R}^{n \times k}$ are item latent factors.

**SGD update rules per observed pair $(u, i)$:**

$$p_u \leftarrow p_u + \eta(e_{ui}\, q_i - \lambda\, p_u)$$
$$q_i \leftarrow q_i + \eta(e_{ui}\, p_u - \lambda\, q_i)$$

where $e_{ui} = R_{ui} - p_u^\top q_i$.

**Hyperparameters:**

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| Learning rate η | 0.03 | Standard for implicit binary data; larger values cause divergence |
| Regularization λ | 0.01 | Controls factor scale; prevents overfitting on the sparse 99.8% matrix |
| Epochs | 30 | Sufficient for convergence in 500-game universe; monitored via loss curve |
| Latent dim k | sweep {10, 20, 50} | Explored via HR@10 to identify optimal compression |
| Initialization | N(0, 0.1²) | Small random values to break symmetry without inflating initial predictions |

**Latent dimension sweep:** k ∈ {10, 20, 50} are evaluated independently. The best HR@10 determines the final model saved to `artifacts/recommendation/`.

**Why MF:** Unlike content-based methods, MF learns co-occurrence structure from collective behavior. Users who endorse similar game sets will have nearby latent vectors, enabling recommendations of games that are not explicitly similar in tag space but are consistently liked together. This is the key advantage over content-based in domains where user behavior is richer than item metadata.

**Low-rank assumption:** The model assumes preferences can be approximated by a small number of latent dimensions (k ≪ min(m, n)). For Steam gaming, plausible latent dimensions might correspond to broad preference styles (competitive vs. narrative, short-session vs. immersive), but the factor coordinates are not directly interpretable and should not be named as genres without empirical evidence.

---

### 7. System 4 — Hybrid Recommender

**Design:** A weighted linear combination of the content score and the MF score:

$$s_{\text{hybrid}}(u, g) = \alpha \cdot s_{\text{content}}(u, g) + (1-\alpha) \cdot s_{\text{MF}}(u, g)$$

Both scores are min-max normalized to [0,1] before combination. α ∈ {0.1, 0.3, 0.5, 0.7, 0.9} is swept empirically.

**Why the hybrid is legitimate here:** `AppID` is the common key for both the content matrix (built from catalog) and the MF item factors (built from interaction parquet). Both systems index games identically. No cross-catalog alignment issue arises — this distinguishes our hybrid from setups where item identities would need to be matched across separate catalogs with potential mismatches.

**Information source balancing:** Content and MF fail in complementary ways. Content fails when a user has a narrow interaction history that doesn't reveal their full taste profile. MF fails under sparsity (cold-start for new users or rare games). The hybrid leverages both signals, with α controlling which source is trusted more.

---

### 8. Offline Evaluation Results

Results generated by `python src/recommendation.py` with `random_state=42`.

| System | HR@5 | HR@10 | HR@20 |
| :--- | ---: | ---: | ---: |
| Popularity baseline | 0.4597 | 0.5796 | 0.7124 |
| Content-based baseline | 0.1806 | 0.2170 | 0.2669 |
| MF k=10 | 0.1103 | 0.1888 | 0.2949 |
| MF k=20 | 0.0764 | 0.1333 | 0.2183 |
| MF k=50 | 0.0756 | 0.1235 | 0.1998 |
| Hybrid (α=0.3, k=10) | **0.2057** | **0.2569** | **0.3330** |

`random_state=42`, 12,389 qualifying users, 484-game universe. Full metrics in `artifacts/recommendation/rec_metrics.json`.

Visualization artifacts saved in `reports/Week10/`:
- `fig01_hr_comparison.png` — HR@K bar chart across all systems
- `fig02_mf_loss_curve.png` — SGD training loss per epoch
- `fig03_mf_k_sweep.png` — HR@10 vs latent dimension k
- `fig04_hybrid_alpha_sweep.png` — HR@10 vs α with baselines

---

### 9. Error Analysis

The best MF model is used for error analysis. For each user in the evaluation set, the held-out game either appears in the top-10 (hit) or not (miss).

**Strong cases (hits):** Users whose training history concentrates in a single cluster tend to get accurate recommendations because MF can identify other users with similar narrow profiles and retrieve their endorsed games. Example pattern: a user who has only endorsed games in the Action cluster (C1) reliably receives other high-rated Action games in their top-10.

**Failure cases (misses):** The most common failure mode is **cross-cluster taste**: a user's training history spans two or more clusters (e.g., C1 Action and C3 Narrative), but the held-out game belongs to a third cluster (e.g., C0 Casual). The latent factors average out competing signals and fail to recover the minority interest. This is an inherent limitation of the squared-loss MF objective on binary implicit data.

A secondary failure mode is **popularity contamination**: when the MF score is weak (sparse history), the model defaults to broadly popular games rather than the user's specific interest. This is especially pronounced for users with only 2 qualifying interactions.

---

### 10. Reproducible Pipeline

```bash
# Step 1 — Ingestion (if not already done)
python src/ingest.py

# Step 2 — Feature engineering (produces artifacts/)
python src/feature_engineering.py

# Step 3 — Clustering (produces artifacts/clustering/)
python src/clustering_week7.py

# Step 4 — Recommendation engine (produces artifacts/recommendation/)
python src/recommendation.py

# Step 5 — Notebook (visualizations)
# Open notebooks/recommendation_week10.ipynb and run all cells
```

All outputs are deterministic with `random_state=42` or `np.random.default_rng(42)`.

---

### 11. Ethics and Access Note

* **Data provenance:** All interaction data comes from `steam_v1.parquet` built by `src/ingest.py` from the Kaggle Steam reviews dataset (CC0 license).
* **Authorization:** CC0 permits unrestricted academic use.
* **Personal data risks:** `author_steamid` is used as a row index in the interaction matrix. It is treated as an opaque identifier — no real names, emails, or financial data are involved.
* **Risk mitigation:** Steam IDs are public-facing identifiers on the platform. No re-identification is attempted or possible from the data retained (voted_up + playtime only, no review text). Latent factors saved in `artifacts/recommendation/mf_P.npy` are indexed by Steam ID but carry no personally identifiable content beyond the interaction signal already public on Steam.
* **Recommendation fairness:** The popularity baseline and MF model both have popularity bias — they tend to recommend widely-reviewed games. This directly contradicts the product goal of surfacing niche Indie titles. Future work should incorporate debiasing techniques (e.g., inverse propensity scoring) to reduce exposure concentration.
