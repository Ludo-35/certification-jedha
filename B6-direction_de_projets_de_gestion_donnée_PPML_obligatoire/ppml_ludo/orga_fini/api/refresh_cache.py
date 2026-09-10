"""
FlyOnTime — Rafraîchissement des tables de référence utilisées par l'API
de prédiction (historique compagnie/aéroport, climatologie météo, trafic
planifié).

Exposé à la fois comme :
- un script exécutable (`python refresh_cache.py`) pour un lancement manuel
- une fonction `refresh()` importable, appelée automatiquement par l'API
  (voir api.py) sur une tâche planifiée quotidienne — utile sur Hugging
  Face Spaces, qui n'a pas de Cron Job natif comme Render.
"""

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
    """Optionnel sur HF Spaces (l'API tourne dans le même conteneur que le
    rafraîchissement, donc le cache local suffit) — gardé pour portabilité
    si le projet est redéployé ailleurs (Render, etc.) avec des conteneurs
    séparés."""
    s3 = boto3.client("s3")
    for fname in CACHE_FILES:
        local_path = f"{cache_dir}/{fname}"
        s3.upload_file(local_path, bucket, f"{prefix}/{fname}")
        print(f"  ↑ {fname} -> s3://{bucket}/{prefix}/{fname}")


def refresh(push_to_s3: bool = False) -> dict:
    """Recalcule le cache et l'écrit sur disque local. Retourne les tables
    de référence (dict) pour un usage direct sans re-lecture disque."""
    print("🔄 Rafraîchissement du cache de features FlyOnTime...")
    df_mouvs = load_data_from_s3(FILES["mouvs"], BUCKET_NAME)
    df_meteo = load_data_from_s3(FILES["meteo"], BUCKET_NAME)
    df_delays = load_data_from_s3(FILES["delays"], BUCKET_NAME)

    ref = build_reference_tables(df_mouvs, df_meteo, df_delays)
    save_reference_tables(ref, CACHE_DIR)
    print(f"✅ Cache rafraîchi localement ({ref['refreshed_at']})")

    if push_to_s3:
        print("📤 Envoi du cache vers S3...")
        upload_cache_to_s3(CACHE_DIR, BUCKET_NAME)
        print("✅ Cache disponible sur S3")

    return ref


if __name__ == "__main__":
    refresh(push_to_s3=True)


