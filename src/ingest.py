import pandas as pd
import glob
import os
import kagglehub
import shutil

def run_ingestion(limit_files=None, sample_frac=0.1, random_state=42):
    print("Starting Automated Big Data Ingestion Pipeline...")

    raw_dir = "data/raw"
    processed_dir = "data/processed"
    metadata_raw_dir = os.path.join(raw_dir, "catalog")
    reviews_raw_dir = os.path.join(raw_dir, "reviews")

    os.makedirs(metadata_raw_dir, exist_ok=True)
    os.makedirs(reviews_raw_dir, exist_ok=True)

    metadata_file = os.path.join(metadata_raw_dir, "games_may2024_full.csv")
    existing_reviews = glob.glob(
        os.path.join(reviews_raw_dir, "**/*.csv"), recursive=True
    )

    if os.path.exists(metadata_file) and existing_reviews:
        print("Using existing raw files in data/raw. Skipping Kaggle download.")
    else:
        # 1. Automated Download from Kaggle
        print("Downloading datasets from Kaggle...")
        catalog_path = kagglehub.dataset_download("artermiloff/steam-games-dataset")
        reviews_path = kagglehub.dataset_download("artermiloff/steam-games-reviews-2024")

        # Save downloaded source files under data/raw for reproducibility
        metadata_source_file = os.path.join(catalog_path, "games_may2024_full.csv")
        if os.path.exists(metadata_source_file):
            shutil.copy2(metadata_source_file, metadata_file)

        source_reviews_files = glob.glob(
            os.path.join(reviews_path, "**/*.csv"), recursive=True
        )
        for src_file in source_reviews_files:
            relative_path = os.path.relpath(src_file, reviews_path)
            dst_file = os.path.join(reviews_raw_dir, relative_path)
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)

    # Process reviews from the raw layer
    raw_reviews_pattern = os.path.join(reviews_raw_dir, "**/*.csv")

    # 2. Inventory of Review Files in raw
    all_files = glob.glob(raw_reviews_pattern, recursive=True)
    print(f"Found {len(all_files)} game review files.")

    # 3. Processing Subset, Stratified Sample, or Full
    if limit_files is None:
        # Stratified sampling by popularity (num_reviews_total)
        games_df = pd.read_csv(metadata_file, usecols=["AppID", "num_reviews_total"])
        games_df["num_reviews_total"] = pd.to_numeric(
            games_df["num_reviews_total"], errors="coerce"
        ).fillna(0)
        games_df["AppID"] = games_df["AppID"].astype(str)

        if games_df["num_reviews_total"].nunique() < 4:
            sampled_games = games_df.sample(frac=sample_frac, random_state=random_state)
            print(
                "⚠️ Not enough unique review counts for quartiles. Using simple random sample."
            )
        else:
            games_df["popularity_bin"] = pd.qcut(
                games_df["num_reviews_total"],
                q=4,
                labels=False,
                duplicates="drop",
            )
            sampled_games = (
                games_df.groupby("popularity_bin", group_keys=False)
                .apply(lambda df: df.sample(frac=sample_frac, random_state=random_state))
                .reset_index(drop=True)
            )
            print(
                "✅ Stratified sampling by popularity (quartiles) applied."
            )

        sampled_appids = set(sampled_games["AppID"].tolist())
        subset_files = [
            f for f in all_files if os.path.basename(f).replace(".csv", "") in sampled_appids
        ]
        print(
            f"🧪 Processing stratified sample: {len(subset_files)} files "
            f"({sample_frac:.0%} per popularity bin)."
        )
    else:
        subset_files = all_files[:limit_files]
        print(f"🧪 Processing a sample of {limit_files} files...")

    list_df = []
    for f in subset_files:
        try:
            # We only read necessary columns to save memory during ingestion
            temp_df = pd.read_csv(f, usecols=['recommendationid', 'author_steamid', 'voted_up', 'author_playtime_forever'])
            appid = os.path.basename(f).replace('.csv', '')
            temp_df['AppID'] = appid
            list_df.append(temp_df)
        except Exception:
            continue

    # 4. Vertical Integration (Concatenation)
    reviews_df = pd.concat(list_df, ignore_index=True)
    
    # 5. Horizontal Integration (Join with Metadata)
    print("Joining with Catalog Metadata...")
    games_df = pd.read_csv(metadata_file, usecols=['AppID', 'name', 'genres', 'tags'])
    
    reviews_df['AppID'] = reviews_df['AppID'].astype(str)
    games_df['AppID'] = games_df['AppID'].astype(str)

    v1_dataset = pd.merge(reviews_df, games_df, on='AppID', how='inner')

    # 6. Storage in Big Data Format (Parquet)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    output_file = (
        f"{processed_dir}/steam_full.parquet"
        if limit_files is None
        else f"{processed_dir}/steam_v1.parquet"
    )
    v1_dataset.to_parquet(output_file, index=False)
    
    print(f"Dataset saved successfully at: {output_file}")
    print(f"📊 Final Record Count: {len(v1_dataset)}")

if __name__ == "__main__":
    run_ingestion()