"""
FlyOnTime — Rafraîchissement quotidien des tables de référence utilisées
par l'API de prédiction (historique compagnie/aéroport, climatologie météo,
trafic planifié).

Sur Render, ce script tourne en Cron Job — un conteneur SÉPARÉ de l'API,
sans disque partagé. Le cache est donc calculé localement PUIS poussé sur
S3 ; l'API le retélécharge à chaque redémarrage (voir sync_cache_from_s3
dans api.py).
"""

import os

import boto3

from engine import load_data_from_s3
from feature_engineering import build_reference_tables, save_reference_tables

BUCKET_NAME = "testludo35"
CACHE_DIR = "cache"
S3_CACHE_PREFIX = "cache"
CACHE_FILES = [
    "airline_table.parquet", "destination_freq.parquet", "airport_stats.parquet",
    "traffic_climo.parquet", "weather_climo.parquet", "meta.json",
]

FILES = {
    "mouvs": "df_mouvs_179jours_20260329_224426.parquet",
    "meteo": "df_meteo_179jours_20260329_224426.parquet",
    "delays": "df_airport_delays_179jours_20260329_224426.parquet",
}


def upload_cache_to_s3(cache_dir: str, bucket: str, prefix: str = S3_CACHE_PREFIX):
    s3 = boto3.client("s3")
    for fname in CACHE_FILES:
        local_path = f"{cache_dir}/{fname}"
        s3.upload_file(local_path, bucket, f"{prefix}/{fname}")
        print(f"  ↑ {fname} -> s3://{bucket}/{prefix}/{fname}")


def main():
    print("🔄 Rafraîchissement du cache de features FlyOnTime...")
    df_mouvs = load_data_from_s3(FILES["mouvs"], BUCKET_NAME)
    df_meteo = load_data_from_s3(FILES["meteo"], BUCKET_NAME)
    df_delays = load_data_from_s3(FILES["delays"], BUCKET_NAME)

    ref = build_reference_tables(df_mouvs, df_meteo, df_delays)
    save_reference_tables(ref, CACHE_DIR)
    print(f"✅ Cache calculé localement ({ref['refreshed_at']})")

    print("📤 Envoi du cache vers S3 (source de vérité partagée avec l'API)...")
    upload_cache_to_s3(CACHE_DIR, BUCKET_NAME)
    print("✅ Cache disponible pour l'API au prochain redémarrage/health-check")


if __name__ == "__main__":
    main()

