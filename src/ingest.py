import glob
import os
import shutil

import kagglehub
import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(ENV_PATH)


def run_ingestion(limit_files=None, sample_frac=1.0, random_state=42):
    print("Starting Automated Big Data Ingestion Pipeline...")

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    processed_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    metadata_raw_dir = os.path.join(raw_dir, "catalog")
    reviews_raw_dir = os.path.join(raw_dir, "reviews")

    os.makedirs(metadata_raw_dir, exist_ok=True)
    os.makedirs(reviews_raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    metadata_file = os.path.join(metadata_raw_dir, "games_may2024_full.csv")
    existing_reviews = glob.glob(os.path.join(reviews_raw_dir, "**/*.csv"), recursive=True)

    if os.path.exists(metadata_file) and existing_reviews:
        print("Using existing raw files in data/raw. Skipping Kaggle download.")
    else:
        missing_env_vars = [
            var_name for var_name in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.getenv(var_name)
        ]
        if missing_env_vars:
            raise EnvironmentError(
                "Missing Kaggle credentials in .env. "
                f"Required variables: {', '.join(missing_env_vars)}. "
                f"Expected file: {ENV_PATH}"
            )

        print("Downloading datasets from Kaggle...")
        catalog_path = kagglehub.dataset_download("artermiloff/steam-games-dataset")
        reviews_path = kagglehub.dataset_download("artermiloff/steam-games-reviews-2024")

        metadata_source_file = os.path.join(catalog_path, "games_may2024_full.csv")
        if not os.path.exists(metadata_source_file):
            raise FileNotFoundError(
                "Downloaded catalog dataset does not contain games_may2024_full.csv."
            )
        shutil.copy2(metadata_source_file, metadata_file)

        source_reviews_files = glob.glob(os.path.join(reviews_path, "**/*.csv"), recursive=True)
        if not source_reviews_files:
            raise FileNotFoundError("No review CSV files were found in the downloaded reviews dataset.")
        for src_file in source_reviews_files:
            relative_path = os.path.relpath(src_file, reviews_path)
            dst_file = os.path.join(reviews_raw_dir, relative_path)
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copy2(src_file, dst_file)

    raw_reviews_pattern = os.path.join(reviews_raw_dir, "**/*.csv")
    all_files = glob.glob(raw_reviews_pattern, recursive=True)
    print(f"Found {len(all_files)} game review files.")
    if not all_files:
        raise RuntimeError("No review CSV files found under data/raw/reviews.")

    if limit_files is not None:
        if limit_files <= 0:
            raise ValueError("limit_files must be a positive integer when provided.")
        subset_files = all_files[:limit_files]
        output_name = "steam_v1.parquet"
        print(f"🧪 Processing a sample of {len(subset_files)} files...")
    elif sample_frac is None or sample_frac == 1:
        subset_files = all_files
        output_name = "steam_full.parquet"
        print("✅ Processing full review dataset.")
    else:
        if sample_frac <= 0 or sample_frac > 1:
            raise ValueError("sample_frac must be in the interval (0, 1] when provided.")

        games_df = pd.read_csv(metadata_file, usecols=["AppID", "num_reviews_total"])
        games_df["num_reviews_total"] = pd.to_numeric(
            games_df["num_reviews_total"], errors="coerce"
        ).fillna(0)
        games_df["AppID"] = games_df["AppID"].astype(str)

        if games_df["num_reviews_total"].nunique() < 4:
            sampled_games = games_df.sample(frac=sample_frac, random_state=random_state)
            print("⚠️ Not enough unique review counts for quartiles. Using simple random sample.")
        else:
            games_df["popularity_bin"] = pd.qcut(
                games_df["num_reviews_total"],
                q=4,
                labels=False,
                duplicates="drop",
            )
            sampled_games = (
                games_df.groupby("popularity_bin", group_keys=False)
                .apply(lambda frame: frame.sample(frac=sample_frac, random_state=random_state))
                .reset_index(drop=True)
            )
            print("✅ Stratified sampling by popularity (quartiles) applied.")

        sampled_appids = set(sampled_games["AppID"].tolist())
        subset_files = [
            f for f in all_files if os.path.basename(f).replace(".csv", "") in sampled_appids
        ]
        output_name = "steam_v1.parquet"
        print(
            f"🧪 Processing stratified sample: {len(subset_files)} files "
            f"({sample_frac:.0%} per popularity bin)."
        )

    if not subset_files:
        raise RuntimeError("No review files selected for ingestion with current parameters.")

    list_df = []
    failed_files = []
    for file_path in subset_files:
        try:
            temp_df = pd.read_csv(
                file_path,
                usecols=["recommendationid", "author_steamid", "voted_up", "author_playtime_forever"],
            )
        except (ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            failed_files.append((file_path, str(exc)))
            continue

        appid = os.path.basename(file_path).replace(".csv", "")
        temp_df["AppID"] = appid
        list_df.append(temp_df)

    if failed_files:
        print(f"⚠️ Skipped {len(failed_files)} unreadable review files.")
        for file_path, error in failed_files[:5]:
            print(f"   - {file_path}: {error}")

    if not list_df:
        raise RuntimeError("No review files could be parsed successfully.")

    reviews_df = pd.concat(list_df, ignore_index=True)

    print("Joining with Catalog Metadata...")
    games_df = pd.read_csv(metadata_file, usecols=["AppID", "name", "genres", "tags"])

    reviews_df["AppID"] = reviews_df["AppID"].astype(str)
    games_df["AppID"] = games_df["AppID"].astype(str)
    final_dataset = pd.merge(reviews_df, games_df, on="AppID", how="inner")

    output_file = os.path.join(processed_dir, output_name)
    final_dataset.to_parquet(output_file, index=False)

    print(f"Dataset saved successfully at: {output_file}")
    print(f"📊 Final Record Count: {len(final_dataset)}")
    return output_file


if __name__ == "__main__":
    run_ingestion()
