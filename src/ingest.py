import pandas as pd
import glob
import os

def run_ingestion(limit_files=500): # Limitamos a 500 archivos para la V1
    print("🚀 Iniciando pipeline de ingesta masiva...")
    
    raw_reviews_path = 'data/raw/SteamReviews2024' 
    metadata_path = 'data/raw/games_may2024_full.csv'   # El archivo de catálogo
    processed_path = 'data/processed'

    # 1. Obtener la lista de todos los archivos de reviews
    all_files = glob.glob(os.path.join(raw_reviews_path, "*.csv"))
    print(f"📂 Se encontraron {len(all_files)} archivos de juegos.")

    # 2. Leer una muestra para la V1
    subset_files = all_files[:limit_files]
    print(f"🧪 Procesando una muestra de {limit_files} archivos...")

    list_df = []
    for f in subset_files:
        try:
            temp_df = pd.read_csv(f)
            appid = os.path.basename(f).replace('.csv', '')
            temp_df['AppID'] = appid
            list_df.append(temp_df)
        except Exception as e:
            continue

    # 3. Concatenar todas las reviews en un solo DataFrame
    reviews_df = pd.concat(list_df, ignore_index=True)
    print(f"✅ Se cargaron {len(reviews_df)} filas de reviews.")

    # 4. Cargar Metadata y unir
    print("📦 Uniendo con metadatos del catálogo...")
    games_df = pd.read_csv(metadata_path)
    
    # Asegurar que el appid sea del mismo tipo para el join
    reviews_df['AppID'] = reviews_df['AppID'].astype(str)
    games_df['AppID'] = games_df['AppID'].astype(str)

    v1_dataset = pd.merge(reviews_df, games_df[['AppID', 'name', 'genres']], on='AppID', how='inner')

    # 5. Guardar en Parquet
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    
    v1_dataset.to_parquet(f'{processed_path}/steam_v1.parquet', index=False)
    print(f"🏁 V1 guardada exitosamente. Total de filas finales: {len(v1_dataset)}")

if __name__ == "__main__":
    run_ingestion()