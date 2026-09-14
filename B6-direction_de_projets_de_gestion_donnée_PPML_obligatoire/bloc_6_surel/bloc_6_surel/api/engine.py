import pandas as pd
import boto3
import io
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# --- PARTIE 1 : DATA ENGINEERING ---

def load_data_from_s3(file_name, bucket_name="testludo35"):
    """Lit un fichier parquet directement à la racine du bucket S3"""
    # Plus besoin de préfixe, on utilise directement le nom du fichier
    s3_path = file_name 
    
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=s3_path)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))
    


def compute_features(df_mouvs, df_meteo, df_delays):
    """Fusionne les données en utilisant les noms de colonnes réels"""
    
    # 1. Normalisation temporelle (Arrondi à l'heure)
    # Pour Mouvs
    df_mouvs['time_h'] = pd.to_datetime(df_mouvs['scheduled_utc']).dt.floor('h')
    
    # Pour Météo
    df_meteo['time_h'] = pd.to_datetime(df_meteo['time']).dt.floor('h')
    
    # Pour Delays
    df_delays['time_h'] = pd.to_datetime(df_delays['window_from_utc']).dt.floor('h')

    # 2. Calcul de la Congestion (Trafic par heure et par aéroport)
    # Note : On utilise 'icao' qui est présent dans Mouvs
    congestion = df_mouvs.groupby(['time_h', 'icao']).size().reset_index(name='traffic_density')

    # 3. Fusion des blocs
    # On commence par Congestion + Météo
    master = pd.merge(congestion, df_meteo, on=['time_h', 'icao'], how='inner')
    
    # On ajoute les Délais (Target)
    # On ne garde que les colonnes utiles de Delays pour éviter de polluer le modèle
    cols_to_keep = ['time_h', 'icao', 'arr_median_delay']
    master = pd.merge(master, df_delays[cols_to_keep], on=['time_h', 'icao'], how='inner')
    
    # Nettoyage final : suppression des colonnes ID ou texte inutiles pour le ML
    # On garde : traffic_density, temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation, cloud_cover
    return master

# --- PARTIE 2 : NEON DB (LOGS) ---

def get_db_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

def save_route_prediction_log(icao_dep, icao_arr, airline, scheduled_utc, dep_delay_minutes, arr_delay_minutes):
    """Enregistre UNE prédiction de trajet complet (décollage + arrivée) dans NeonDB."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            query = """
            INSERT INTO prediction_logs (icao_dep, icao_arr, airline, scheduled_utc, dep_delay_minutes, arr_delay_minutes)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            cur.execute(query, (icao_dep, icao_arr, airline, scheduled_utc, dep_delay_minutes, arr_delay_minutes))
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"Erreur NeonDB : {e}")

def fetch_recent_logs(limit=5):
    """Récupère l'historique pour l'app"""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT timestamp, icao_dep, icao_arr, airline, scheduled_utc, dep_delay_minutes, arr_delay_minutes "
                "FROM prediction_logs ORDER BY timestamp DESC LIMIT %s",
                (limit,),
            )
            return cur.fetchall()
    except:
        return []