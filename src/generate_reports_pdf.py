"""Generate Technical_report_G6_W5.pdf and Technical_report_G6_W10.pdf using fpdf2."""
from __future__ import annotations

import json
import os

from fpdf import FPDF, XPos, YPos

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W5_DIR  = os.path.join(PROJECT_ROOT, "reports", "Week5")
W10_DIR = os.path.join(PROJECT_ROOT, "reports", "Week10")
ART_DIR = os.path.join(PROJECT_ROOT, "artifacts")

AUTHORS = (
    "Iam Anthony Marcelo Alvarez Orellana  |  Jeffrey Ulises Diaz Villanueva\n"
    "Paula Jimena Mancilla Cienfuegos  |  Fernando Samuel Paredes Espinoza"
)
HEADER_TEXT = "Semester Project: Domain Discovery, Recommendation, and Graph Intelligence"
LM = 20
RM = 20
PW = 210 - LM - RM  # 170mm usable width


class ReportPDF(FPDF):
    def __init__(self, subtitle: str):
        super().__init__()
        self.subtitle = subtitle
        self.set_auto_page_break(auto=True, margin=20)
        self.set_left_margin(LM)
        self.set_right_margin(RM)

    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(100, 100, 100)
        self.cell(PW, 6, HEADER_TEXT, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_draw_color(180, 180, 180)
        self.line(LM, self.get_y(), 210 - RM, self.get_y())
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(PW, 6, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def title_page(self):
        self.add_page()
        self.ln(20)
        self.set_font("Helvetica", "B", 16)
        self.multi_cell(PW, 10, HEADER_TEXT, align="C")
        self.ln(6)
        self.set_font("Helvetica", "", 13)
        self.multi_cell(PW, 8, self.subtitle, align="C")
        self.ln(10)
        self.set_draw_color(60, 100, 180)
        self.set_line_width(0.5)
        self.line(40, self.get_y(), 170, self.get_y())
        self.ln(10)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(60, 60, 60)
        self.multi_cell(PW, 7, AUTHORS, align="C")
        self.set_text_color(0, 0, 0)
        self.set_line_width(0.2)

    def section(self, num: int, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(230, 238, 255)
        self.cell(PW, 8, f"{num}. {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        self.ln(2)

    def subsection(self, title: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.cell(PW, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10.5)
        self.multi_cell(PW, 6, text)
        self.ln(1)

    def bullet(self, items: list[str]):
        self.set_font("Helvetica", "", 10.5)
        indent = 6
        for item in items:
            self.set_x(LM + indent)
            self.multi_cell(PW - indent, 6, f"- {item}")
        self.ln(1)

    def fig(self, path: str, caption: str, w: float = 160):
        if not os.path.exists(path):
            self.body(f"[Figure not found: {os.path.basename(path)}]")
            return
        self.ln(2)
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(PW, 5, f"Figure: {caption}", align="C")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def table(self, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None):
        n = len(headers)
        if col_widths is None:
            col_widths = [PW / n] * n

        # Header row
        self.set_font("Helvetica", "B", 9.5)
        self.set_fill_color(50, 100, 180)
        self.set_text_color(255, 255, 255)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, h, border=1, fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)

        # Data rows
        self.set_font("Helvetica", "", 9.5)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            self.set_fill_color(240, 244, 255) if fill else self.set_fill_color(255, 255, 255)

            # Compute row height (max lines across all cells)
            max_lines = 1
            for cell_text, cw in zip(row, col_widths):
                # Temporarily compute line count
                lines = len(self.multi_cell(cw, 6, str(cell_text), border=0, fill=False, split_only=True))
                max_lines = max(max_lines, lines)
            row_h = max_lines * 6

            x0 = self.get_x()
            y0 = self.get_y()
            for cell_text, cw in zip(row, col_widths):
                self.set_xy(x0, y0)
                self.multi_cell(cw, 6, str(cell_text), border=1, fill=fill)
                x0 += cw
            self.set_xy(LM, y0 + row_h)
        self.ln(3)

    def code(self, text: str):
        self.set_font("Courier", "", 9)
        self.set_fill_color(245, 245, 245)
        self.multi_cell(PW, 5, text, fill=True, border=1)
        self.ln(2)
        self.set_font("Helvetica", "", 10.5)


# ---------------------------------------------------------------------------
# Week 5
# ---------------------------------------------------------------------------

def build_w5():
    pdf = ReportPDF("Milestone: Feature Engineering and Dimensionality Reduction (Week 5)")
    pdf.title_page()
    pdf.add_page()

    pdf.section(1, "Overview")
    pdf.body(
        "This milestone covers the construction of the three feature matrices used as input to "
        "all downstream models (clustering, recommendation, graph analysis), and the dimensionality "
        "reduction analysis used to select compression parameters.\n"
        "Code: src/feature_engineering.py  |  Artifacts: artifacts/  |  Figures: src/generate_figures.py"
    )

    pdf.section(2, "Dataset Snapshot")
    pdf.bullet([
        "Processed dataset: data/processed/steam_v1.parquet  -  480,025 interaction records",
        "Games with tag metadata (clustering): 69,943 titles",
        "Games in feature matrices: 87,806 titles (full catalog)",
    ])

    pdf.section(3, "Feature Matrices Built")
    pdf.table(
        ["Matrix", "Shape", "Type", "Content"],
        [
            ["X_numeric",     "87,806 x 26",   "Dense",      "Numerical catalog fields. StandardScaler applied."],
            ["X_categorical", "87,806 x 5,000", "Sparse CSR", "Top-5,000 genre/category tokens. Binary encoding."],
            ["X_text",        "87,806 x 5,000", "Sparse CSR", "TF-IDF on name + description + tags (1-2 grams)."],
        ],
        col_widths=[30, 28, 24, 88],
    )
    pdf.subsection("Numeric columns (26)")
    pdf.body(
        "Required age, price, DLC count, Metacritic score, achievements, recommendations, user score, "
        "score rank, positive/negative counts, estimated owners, average/median playtime (forever and "
        "2-week windows), peak CCU, pct_pos (total/recent), num_reviews (total/recent), "
        "Windows/Mac/Linux flags, release year, release month, release age in days."
    )

    pdf.section(4, "Dimensionality Reduction Analysis")
    pdf.subsection("PCA on X_numeric")
    pdf.bullet([
        "Components tested: up to 26 (full rank).",
        "Result: k=12 components capture >= 90% of variance.",
        "12 orthogonal directions cover price tiers, engagement levels, platform availability, recency.",
    ])
    pdf.fig(
        os.path.join(W5_DIR, "figs", "fig_w5_01_pca_variance.png"),
        "PCA on X_numeric: cumulative explained variance. Red dashed = 90% threshold; orange dotted = k=12.",
        w=150,
    )

    pdf.subsection("TruncatedSVD on X_text")
    pdf.bullet([
        "Components tested: up to 300.",
        "Result: k=201 components capture >= 50% of total squared singular value mass.",
        "Slow singular value decay (no sharp elbow) is typical of broad-vocabulary document collections.",
    ])
    pdf.fig(
        os.path.join(W5_DIR, "figs", "fig_w5_02_svd_spectrum.png"),
        "TruncatedSVD on X_text: singular value spectrum (left) and cumulative variance (right).",
        w=165,
    )

    pdf.subsection("Summary")
    pdf.table(
        ["Matrix", "Original dims", "Reduced dims", "Criterion", "Method"],
        [
            ["X_numeric",     "26",    "12",         ">= 90% variance", "PCA"],
            ["X_text",        "5,000", "201",         ">= 50% variance", "TruncatedSVD"],
            ["X_categorical", "5,000", "50 (fixed)", "downstream use",  "TruncatedSVD"],
        ],
        col_widths=[32, 28, 28, 44, 38],
    )

    pdf.section(5, "Feature Matrix Dimensions")
    pdf.fig(
        os.path.join(W5_DIR, "figs", "fig_w5_03_matrix_dims.png"),
        "Feature column count per matrix. X_numeric is dense; X_categorical and X_text are sparse.",
        w=130,
    )

    pdf.section(6, "Reproducible Pipeline")
    pdf.code("# From project root\npython src/feature_engineering.py")
    pdf.body(
        "Artifacts: X_numeric.npz, X_categorical.npz, X_text.npz, models.pkl, metrics.json\n"
        "All outputs deterministic (random_state=42)."
    )

    pdf.section(7, "Ethics and Access Note")
    pdf.bullet([
        "Provenance: Feature matrices derived from steam_v1.parquet. No review text or PII.",
        "Authorization: CC0 license; unrestricted academic use.",
        "Bias note: X_numeric includes popularity counts introducing popularity bias (addressed in W10).",
    ])

    out = os.path.join(W5_DIR, "Technical_report_G6_W5.pdf")
    pdf.output(out)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Week 10
# ---------------------------------------------------------------------------

def build_w10():
    with open(os.path.join(ART_DIR, "recommendation", "rec_metrics.json")) as f:
        metrics = json.load(f)
    ev = metrics["evaluation"]

    pdf = ReportPDF("Milestone: Recommendation, Ranking, and Predictive Decision Engine (Week 10)")
    pdf.title_page()
    pdf.add_page()

    pdf.section(1, "Task Framing")
    pdf.body(
        "Task type: Recommendation.\n\n"
        "Given a user's positive interaction history (games endorsed via voted_up=True), "
        "the system predicts which unseen game they would also endorse. Output: ranked list.\n\n"
        "Clustering results from Week 7 (6 segments: Casual, Action, Local Multiplayer, "
        "Narrative/JRPG, VR, Strategy) feed into this pipeline as candidate-filtering features."
    )

    pdf.section(2, "Data Alignment")
    pdf.body("All systems operate on the same 500-game universe from data/processed/steam_v1.parquet. AppID is the common join key:")
    pdf.bullet([
        "steam_v1.parquet  -  user records (author_steamid, AppID, voted_up, playtime)",
        "games_may2024_full.csv  -  game metadata (name, tags, genres, categories)",
        "artifacts/clustering/cluster_labels_k6.npy  -  cluster labels from Week 7",
    ])

    pdf.section(3, "Evaluation Protocol")
    pdf.bullet([
        "Interaction matrix: sparse binary, users x games. voted_up=True -> 1.",
        "Qualifying users: >= 2 distinct positive interactions.",
        "Train/test split: one positive per user held out (random_state=42).",
        "Primary metric: HR@10 - fraction of users whose held-out game is in top-10.",
        "Leakage prevention: training items masked to -inf before ranking.",
    ])
    pdf.table(
        ["Statistic", "Value"],
        [
            ["Positive interactions",    "442,652"],
            ["Unique users",             "426,601"],
            ["Unique games (universe)",  "484"],
            ["Qualifying users (eval)",  "12,389"],
            ["Matrix density",           "0.0078%"],
        ],
        col_widths=[85, 85],
    )

    pdf.section(4, "Baseline 1 - Popularity")
    pdf.body(
        "Design: All games ranked by aggregate 'recommendations' count. Same ranking for every user.\n"
        "Justification: Non-personalized ceiling. Any personalized system failing to beat it is unjustified.\n"
        "Limitation: Biases toward AAA titles; cannot surface niche Indie games."
    )

    pdf.section(5, "Baseline 2 - Content-Based")
    pdf.body(
        "TF-IDF on name + description + tags + genres + categories (1-2 grams, 3,000 features). "
        "TruncatedSVD -> 50D. User vector = centroid of training game vectors. Ranking by cosine similarity.\n"
        "Limitation: Overspecializes to known preferences; ignores collective behavioral signal."
    )

    pdf.section(6, "System 3 - Matrix Factorization (SGD)")
    pdf.body(
        "Objective:\n"
        "  min_{P,Q} sum_{(u,i)} (R_ui - p_u^T q_i)^2 + lambda*(||P||^2 + ||Q||^2)\n\n"
        "SGD updates per pair (u,i):\n"
        "  p_u <- p_u + eta*(e_ui*q_i - lambda*p_u)\n"
        "  q_i <- q_i + eta*(e_ui*p_u - lambda*q_i)\n"
        "  where e_ui = R_ui - p_u^T q_i"
    )
    pdf.table(
        ["Parameter", "Value", "Justification"],
        [
            ["Learning rate eta",     "0.03",      "Standard for implicit binary data"],
            ["Regularization lambda", "0.01",      "Prevents overfitting on 99.8% sparse matrix"],
            ["Epochs",                "30",         "Sufficient for convergence in 500-game universe"],
            ["Latent dim k",          "{10,20,50}", "Swept via HR@10; best k=10 used for final model"],
            ["Init",                  "N(0,0.01)", "Small random values to break symmetry"],
        ],
        col_widths=[45, 25, 100],
    )

    pdf.section(7, "System 4 - Hybrid Recommender")
    pdf.body(
        "s_hybrid(u,g) = alpha * s_content(u,g) + (1-alpha) * s_MF(u,g)\n"
        "Both scores min-max normalized to [0,1]. alpha in {0.1, 0.3, 0.5, 0.7, 0.9}.\n\n"
        "Hybrid is legitimate: AppID is the common key for both content matrix and MF item factors. "
        "Both systems index games identically - no cross-catalog alignment issue."
    )

    pdf.section(8, "Offline Evaluation Results")
    pdf.body("Generated by python src/recommendation.py with random_state=42.")
    pdf.table(
        ["System", "HR@5", "HR@10", "HR@20"],
        [
            ["Popularity baseline",             "0.4597", "0.5796", "0.7124"],
            ["Content-based baseline",          "0.1806", "0.2170", "0.2669"],
            ["MF k=10",                         "0.1103", "0.1888", "0.2949"],
            ["MF k=20",                         "0.0764", "0.1333", "0.2183"],
            ["MF k=50",                         "0.0756", "0.1235", "0.1998"],
            ["Hybrid (alpha=0.3, k=10)  BEST",  "0.2057", "0.2569", "0.3330"],
        ],
        col_widths=[75, 32, 32, 31],
    )

    pdf.fig(
        os.path.join(W10_DIR, "fig01_hr_comparison.png"),
        "HR@K comparison. Hybrid (alpha=0.3) achieves best personalized HR@10=25.7%.",
        w=165,
    )

    pdf.add_page()

    pdf.fig(
        os.path.join(W10_DIR, "fig02_mf_loss_curve.png"),
        "SGD loss per epoch (k=10). Monotonic decrease confirms convergence within 30 epochs.",
        w=140,
    )
    pdf.fig(
        os.path.join(W10_DIR, "fig03_mf_k_sweep.png"),
        "HR@10 vs latent dimension k. k=10 generalizes best on this sparse 484-game universe.",
        w=130,
    )
    pdf.fig(
        os.path.join(W10_DIR, "fig04_hybrid_alpha_sweep.png"),
        "Hybrid HR@10 vs alpha. Optimal alpha=0.3 (30% content + 70% MF).",
        w=130,
    )

    pdf.section(9, "Error Analysis")
    pdf.body(
        "Strong cases (hits): Users with history concentrated in one cluster get accurate recs. "
        "MF identifies similar narrow-profile users.\n\n"
        "Failure cases (misses): Cross-cluster taste - history spans clusters (e.g., Action + Narrative) "
        "but held-out game belongs to a third cluster. Latent factors average out competing signals.\n\n"
        "Secondary failure: Popularity contamination in users with only 2 qualifying interactions."
    )

    pdf.section(10, "Reproducible Pipeline")
    pdf.code(
        "python src/ingest.py              # Step 1: data ingestion\n"
        "python src/feature_engineering.py # Step 2: feature matrices\n"
        "python src/clustering_week7.py    # Step 3: clustering (Week 7)\n"
        "python src/recommendation.py      # Step 4: recommendation engine\n"
        "python src/generate_figures.py    # Step 5: report figures"
    )
    pdf.body("All outputs deterministic with random_state=42.")

    pdf.section(11, "Ethics and Access Note")
    pdf.bullet([
        "CC0 license. No PII beyond public Steam IDs used as opaque row indices.",
        "author_steamid: opaque identifier. No real names, emails, or financial data.",
        "Recommendation fairness: popularity bias present. Future: inverse propensity scoring.",
    ])

    out = os.path.join(W10_DIR, "Technical_report_G6_W10.pdf")
    pdf.output(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    build_w5()
    build_w10()
    print("Done.")
