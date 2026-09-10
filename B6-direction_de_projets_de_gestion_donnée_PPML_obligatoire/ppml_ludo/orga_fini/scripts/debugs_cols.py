from engine import load_data_from_s3
import pandas as pd

for key, name in {
    "Mouvs": "df_mouvs_179jours_20260329_224426.parquet",
    "Météo": "df_meteo_179jours_20260329_224426.parquet",
    "Delays": "df_airport_delays_179jours_20260329_224426.parquet"
}.items():
    df = load_data_from_s3(name)
    print(f"\n--- Colonnes dans {key} ---")
    print(df.columns.tolist())