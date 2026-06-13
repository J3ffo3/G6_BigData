# Semester Project: Domain Discovery, Recommendation, and Graph Intelligence
## Milestone: Clustering and Validation Report (Week 7)

---

### Team 6 Members

| # | Team Member |
| --- | --- |
| 1 | Iam Anthony Marcelo Alvarez Orellana |
| 2 | Jeffrey Ulises Diaz Villanueva |
| 3 | Paula Jimena Mancilla Cienfuegos |
| 4 | Fernando Samuel Paredes Espinoza |

---

### 1. Objective

Segment the Steam game catalog into meaningful groups and validate whether those groups correspond to genuine domain-level differences in game identity. The segmentation feeds directly into the Week 10 recommendation pipeline as a candidate-filtering mechanism.

---

### 2. Representation Choice and Motivation

**Input:** Game catalog (`games_may2024_full.csv`, ~88k titles). After filtering games without any community tag, **87,806 games** remain.

**Why cluster on tags, not on commercial or numeric features?**

The product question asks how to surface niche Indie titles buried in the 88k-game catalog. Clustering by price or popularity would group a USD 60 AAA title with a USD 60 Indie simply by commercial magnitude — useless for discovery. Clustering by community tags places that Indie roguelike next to *Hades* and *Dead Cells*, which is precisely where a player would expect to find it.

**Feature construction pipeline:**

1. **Columns used:** `genres`, `categories`, `tags`. Developers, publishers, and language fields are excluded: a game should not resemble another merely because both support Simplified Chinese.
2. **Tokenization:** `normalize_multilabel` strips list/dict formatting from stored strings. Each token is kept as-is (no field prefix, since all three columns describe game identity).
3. **Frequency filter:** tokens appearing in fewer than 30 games are removed → **~493 surviving tag dimensions**.
4. **MultiLabelBinarizer** produces a sparse CSR binary matrix of shape `(87,806 × ~493)`.
5. **TruncatedSVD** reduces to **50 latent components** (~62% cumulative explained variance).
6. **L2 normalization** projects every game onto the unit sphere so that K-means distance is a monotone function of cosine similarity, preventing heavily-tagged games from dominating by norm magnitude.

---

### 3. Lloyd's Algorithm — Manual Descent Verification

To verify K-means' monotonic-descent property empirically, one full iteration (assignment → update → re-assignment) was run from five distinct random seeds on `PCA(X_numeric, 10D)` with K = 6.

| Seed | J before | J after | ΔJ | Descent OK |
| ---: | ---: | ---: | ---: | :---: |
| 1 | 59,335.6 | 42,933.7 | −16,402 | Yes |
| 2 | 59,448.2 | 47,519.6 | −11,929 | Yes |
| 3 | 56,224.3 | 42,521.8 | −13,702 | Yes |
| 4 | 51,373.5 | 39,969.5 | −11,404 | Yes |
| 5 | 56,081.7 | 42,075.2 | −14,007 | Yes |

All five seeds satisfy J(after) ≤ J(before), confirming Lloyd's monotonicity. The variance across seeds (ΔJ ranges from −11,404 to −16,402) illustrates why `n_init=10` is used in the final run: different initializations reach different local optima, and retaining the best is necessary.

For reference, scikit-learn run to full convergence with `n_init=10` reaches SSE ≈ 35,200 — consistent with the continued decrease beyond the one hand-computed step.

---

### 4. Ablation Study — Justifying SVD + L2

Three variants of the same clustering were compared on K = 6 to validate the representation design choice:

| Space | Silhouette | Largest cluster share |
| :--- | ---: | ---: |
| Raw sparse (binary tags) | 0.001 | 22.7% |
| SVD only (50D dense) | 0.036 | 21.9% |
| **SVD + L2 (chosen)** | **0.069** | 25.3% |

**Interpretation:**

The L2 step alone produces a ~70× improvement in silhouette over the raw representation. In domain terms: without L2, a game with 30 tags would have a norm of √30 ≈ 5.5, while one with 5 tags would have norm ≈ 2.2. K-means would cluster them together not because of shared type, but because of similar annotation density. After L2, both live on the unit sphere and are grouped by the direction of their tag vectors, which encodes what kind of game they are, not how thoroughly they were tagged.

The choice of SVD + L2 is therefore principled and measured, not aesthetic.

---

### 5. Parameter Sweep — K Selection

K-means was run for K ∈ {4, 6, 8, 10} on the SVD + L2 representation, with `n_init=10`, `random_state=42`, `init='k-means++'`.

| K | Inertia | Silhouette | Min cluster size | Max cluster size |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 37,476 | 0.0643 | 12,655 | 21,324 |
| **6** | **35,319** | **0.0676** | 4,793 | 18,932 |
| 8 | 33,800 | 0.0655 | 4,649 | 13,220 |
| 10 | 32,621 | 0.0676 | 4,325 | 8,650 |

**K = 6 was selected** based on three criteria:
1. Highest silhouette in the bottom-knee region (tied with K=10, but K=6 is more compact).
2. No degenerate clusters (min size 4,793 — large enough to support content-based similarity).
3. Direct interpretability against Steam's observable market segments.

K = 10 achieves the same silhouette but fragments clusters that are semantically coherent (e.g., splitting Action into sub-genres that do not yet have sufficient signal to be independently stable).

---

### 6. Cluster Profiles — The Six Segments

Final clustering: `KMeans(n_clusters=6, n_init=10, random_state=42, init='k-means++')` on SVD + L2.

| Cluster | Size | Share | Identity (top tags by lift) |
| :--- | ---: | ---: | :--- |
| C0 | ~13,586 | 19% | **Casual / Puzzle** — match-3, logic, clicker, relaxing, hidden object |
| C1 | ~18,742 | 27% | **Action** — action (96%), roguelike, bullet hell, platformer |
| C2 | ~10,674 | 15% | **Local Multiplayer** — shared screen, local co-op (lift 6×) |
| C3 | ~13,145 | 19% | **Narrative / JRPG** — RPGMaker, visual novel, interactive fiction |
| C4 | ~6,000 | 9% | **VR** — VR exclusive (88%), tracked controllers (84%) |
| C5 | ~7,800 | 11% | **Strategy** — grand strategy, RTS, city builder, management |

Cluster stability was evaluated with the Adjusted Rand Index (ARI) across five independent seeds. Mean off-diagonal ARI > 0.8, indicating the partition is consistent and not an artifact of initialization.

---

### 7. Failure Analysis

K-means has structural limitations that surface in this domain:

- **Hybrid games at boundaries:** A "souls-like roguelike" is forced entirely into C1 (Action). Its individual silhouette coefficient is close to zero — it genuinely belongs to a fuzzy region between C1 and C3. The algorithm cannot express partial membership.
- **Variable density:** C1 (Action, ~18,700 games) is an amorphous mass of FPS, metroidvania, and bullet hell titles. K-means cannot subdivide it further without producing degenerate clusters at K=8, because these sub-genres share enough vocabulary to sit close in tag space.
- **Asset-flip contamination:** A subset of Steam's catalog consists of low-effort games with copied or generic tag sets. These distort centroids by pulling them toward the mean tag distribution rather than toward a coherent game identity. K-means has no rejection mechanism for outliers.
- **K is exogenous:** The silhouette score is a heuristic, not a proof. K = 6 is the best defensible choice given the evidence, but the "true" number of Steam segments is unknown and likely not fixed.

These limitations motivate the Week 10 recommendation approach: cluster labels are used as a candidate-filtering feature, not as the sole source of similarity.

---

### 8. Reproducible Pipeline

```bash
# 1. Run feature engineering first (produces artifacts needed by clustering)
python src/feature_engineering.py

# 2. Run clustering pipeline
python src/clustering_week7.py
```

Outputs written to `artifacts/clustering/`:
- `cluster_labels_k6.npy` — integer cluster label per tagged game (87,806 entries)
- `cluster_appids.npy` — AppID array aligned to the labels
- `clustering_metrics.json` — all numeric results (Lloyd's demo, ablation, sweep, ARI)
- `cluster_profiles.json` — top-5 tags by lift per cluster

Full technical report (PDF): `reports/Week7/Technical_report_G6_W7.pdf`

---

### 9. Ethics and Access Note

* **Data provenance:** All experiments operate on `games_may2024_full.csv` (CC0 license, Kaggle). No user data is involved in clustering.
* **Authorization:** CC0 permits unrestricted academic use.
* **Personal data risks:** This milestone processes only game metadata. No `author_steamid` or personal identifiers are present in any clustering input or output.
* **Risk mitigation:** Cluster labels are saved per game, not per user. No inference is made about individual player behavior at this stage.
