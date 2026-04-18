import pandas as pd
import glob
import os
import kagglehub
import shutil

def run_ingestion(limit_files=500):
    print("Starting Automated Big Data Ingestion Pipeline...")

    # 1. Automated Download from Kaggle
    print("Downloading datasets from Kaggle...")
    catalog_path = kagglehub.dataset_download("artermiloff/steam-games-dataset")
    reviews_path = kagglehub.dataset_download("artermiloff/steam-games-reviews-2024")
    
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    metadata_raw_dir = os.path.join(raw_dir, 'catalog')
    reviews_raw_dir = os.path.join(raw_dir, 'reviews')

    if not os.path.exists(metadata_raw_dir):
        os.makedirs(metadata_raw_dir)
    if not os.path.exists(reviews_raw_dir):
        os.makedirs(reviews_raw_dir)

    # Save downloaded source files under data/raw for reproducibility
    metadata_source_file = os.path.join(catalog_path, 'games_may2024_full.csv')
    metadata_file = os.path.join(metadata_raw_dir, 'games_may2024_full.csv')
    if os.path.exists(metadata_source_file):
        shutil.copy2(metadata_source_file, metadata_file)

    source_reviews_files = glob.glob(os.path.join(reviews_path, "**/*.csv"), recursive=True)
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

    # 3. Processing V1 Subset
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
    
    output_file = f'{processed_dir}/steam_v1.parquet'
    v1_dataset.to_parquet(output_file, index=False)
    
    print(f"V1 saved successfully at: {output_file}")
    print(f"📊 Final Record Count: {len(v1_dataset)}")

if __name__ == "__main__":
    run_ingestion()