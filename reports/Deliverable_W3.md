# Semester Project: Domain Discovery, Recommendation, and Graph Intelligence
## Milestone: Dataset Charter and Processed Dataset V1 (Week 3)

---

### 1. Project Proposal

* **Domain:** Global Digital Video Game Marketplace (Steam).
* **Problem Statement:** With over 80,000 titles available, the Steam platform faces an acute **"Choice Overload"** problem. Users typically gravitate toward high-budget (AAA) titles, while high-quality niche (Indie) games remain undiscovered. Traditional popularity-based ranking systems fail to capture latent connections between complex gameplay preferences and community-driven structures.
* **Expected Product Question:** How can we improve game discovery through a hybrid system that leverages community-driven tag centrality within a graph and historical user behavior (playtime) to recommend titles with high satisfaction probability?
* **Suitability for the course:** This dataset is ideal for building the four mandatory course layers:
    * **Catalog Layer:** Technical and commercial game metadata (genres, developers, tags).
    * **Feature Layer:** Vectorization of game descriptions and community tags for similarity analysis.
    * **Interaction Layer:** A dataset of ~480k unique interactions including playtime and voting behavior.
    * **Graph Layer:** Bipartite User-Game graphs and Tag-to-Tag co-occurrence networks.

---

### 2. Source Inventory

* **Source URLs:** [Steam Games Dataset 2025 - Artemiy Ermilov (Kaggle)](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset).
[Steam Games Reviews 2024/2025 - Artemiy Ermilov (Kaggle)](https://www.kaggle.com/datasets/artermiloff/steam-games-reviews-2024).
* **Licenses:** CC0: Public Domain / Kaggle Dataset Terms of Use.
* **Raw File Formats:** CSV (multiple files for reviews) and a single CSV for the game catalog.
* **Estimated Size:** Approximately 128 million total records in the universe, sampled down for the V1 working subset.

---

### 3. Schema Draft

The system follows a star schema architecture where `AppID` serves as the central join key between the catalog and the interaction layers.



* **Main Entities:**
    * **Games (Catalog):** Technical, commercial, and descriptive metadata.
    * **Reviews (Interactions):** User sentiment, engagement metrics, and behavioral data.
* **Keys:**
    * `AppID`: Primary Key for Games table; Foreign Key for Reviews.
    * `author_steamid`: Unique identifier for users (Author).
    * `recommendationid`: Unique identifier for each interaction/review record.
* **Expected Joins:** Inner join on `AppID` to enrich interaction data with genres, tags, and developer information.

---

### 4. Data Dictionary Draft (Core Features)

#### Table: Games (Catalog)
| Column | Type | Description |
| :--- | :--- | :--- |
| `AppID` | Integer | Unique identifier for the game on the Steam platform. |
| `name` | String | Official title of the videogame. |
| `genres` | String | Primary categories (e.g., Action, RPG, Strategy). |
| `tags` | String | Community-driven descriptors used for graph construction. |
| `price` | Float | Current retail price in USD. |
| `average_playtime_forever` | Integer | Average playtime across all owners (Global baseline). |

#### Table: Reviews (Interactions)
| Column | Type | Description |
| :--- | :--- | :--- |
| `recommendationid` | Integer | Unique ID for the specific review record. |
| `author_steamid` | String | Anonymized unique ID of the user. |
| `voted_up` | Boolean | Target Variable: Indicates if the user recommends the game. |
| `author_playtime_forever` | Float | Total hours played by the specific user (Engagement metric). |
| `author_num_games_owned` | Integer | User profile feature representing library depth. |

---

### 5. Scale Analysis (V1)

* **Total Raw Files Found:** 79,994 files.
* **V1 Working Subset:** 500 game files processed.
* **Rows in V1:** 480,025 unique interaction records.
* **Columns:** 67 original columns reduced to 12 prioritized features in the pipeline.
* **Sparsity Estimate:** ~99.8% (standard for large-scale recommendation domains).
* **Memory Footprint:** ~115 MB in optimized Parquet format.

---

### 6. Ethics and Access Note

* **Provenance:** Data is sourced from the public Steamworks API and structured by researchers for academic purposes.
* **Authorization:** Data use is permitted under the Steam API Terms of Service for research and non-commercial development.
* **Personal Data Risks:** The dataset includes `author_steamid`. While these are public profiles, there is a risk of profile tracking.
* **Risk Mitigation:** The ingestion pipeline excludes free-text comments to prevent Personal Identifiable Information (PII) leakage. IDs are treated as opaque numerical identifiers. No emails, real names, or financial data are collected or stored.

---

### 7. Technical Expectation: Ingestion Command

To reproduce the processed V1 dataset, run the following command from the project root:

```bash
# 1. Activate virtual environment
# Windows: .\venv\Scripts\activate | Mac/Linux: source venv/bin/activate

# 2. Run ingestion pipeline
python src/ingest.py