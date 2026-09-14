"""
FlyOnTime — Feature engineering pour la prédiction du retard (en minutes)
des vols commerciaux, au départ ET à l'arrivée, à un horizon de 72h.

PRINCIPE DIRECTEUR (à ne jamais casser) :
Toute feature doit être calculable au moment T = scheduled_utc - 72h.
=> INTERDITS comme features : revised_utc, runway_utc, status
   (connus seulement au moment du vol ou juste avant).
=> Les statistiques d'historique (compagnie, aéroport) sont calculées avec
   un décalage (HORIZON_DAYS) pour ne jamais regarder dans les 72h qui
   précèdent le vol.
"""

import numpy as np
import pandas as pd

HORIZON_DAYS = 3          # 72h -> on arrondit à 3 jours pour les agrégats journaliers
LOOKBACK_DAYS = 30        # fenêtre glissante pour les stats d'historique
RESOLVED_STATUSES = ["Departed", "Arrived"]

# Aéroports couverts par le modèle (les 5 plus grands aéroports français
# du dataset d'entraînement). Toute requête hors de ce périmètre doit être
# rejetée explicitement plutôt que de recevoir une prédiction dégradée.
COVERED_AIRPORTS = ["LFPG", "LFPO", "LFLL", "LFMN", "LFML"]

# Vacances scolaires France (zones A/B/C), 2025-2026 — période couverte par les données.
# Format : (date_debut, date_fin) ; on ne distingue pas les zones ici (approximation
# raisonnable pour un premier modèle : impact national du trafic aérien).
FRENCH_SCHOOL_HOLIDAYS_2025_2026 = [
    ("2025-10-18", "2025-11-03"),  # Toussaint
    ("2025-12-20", "2026-01-05"),  # Noël
    ("2026-02-07", "2026-03-09"),  # Hiver (zones A/B/C étalées)
    ("2026-04-04", "2026-05-04"),  # Printemps (zones A/B/C étalées)
]


# ------------------------------------------------------------------
# 1. Chargement / nettoyage de la base vols (df_mouvs)
# ------------------------------------------------------------------
def load_and_clean_mouvs(df_mouvs: pd.DataFrame) -> pd.DataFrame:
    """Ne garde que les vols avec une vérité terrain fiable (statut résolu,
    delay_minutes renseigné), et retire les valeurs aberrantes extrêmes."""
    df = df_mouvs.copy()
    df["scheduled_utc"] = pd.to_datetime(df["scheduled_utc"])

    df = df[df["status"].isin(RESOLVED_STATUSES)]
    df = df.dropna(subset=["delay_minutes"])

    # Bornage : au-delà de -60 / +720 min, ce sont des cas très marginaux
    # (avance improbable, incident majeur) qui bruitent plus qu'ils n'aident.
    before = len(df)
    df = df[df["delay_minutes"].between(-60, 720)]
    removed = before - len(df)

    df = df.sort_values("scheduled_utc").reset_index(drop=True)
    print(f"[clean] {len(df):,} vols conservés (statut résolu + delay connu), "
          f"{removed} valeurs aberrantes retirées")
    return df


# ------------------------------------------------------------------
# 2. Features calendaires (100% safe : dérivées de scheduled_utc)
# ------------------------------------------------------------------
def _is_school_holiday(ts: pd.Timestamp) -> bool:
    d = ts.date()
    for start, end in FRENCH_SCHOOL_HOLIDAYS_2025_2026:
        if pd.Timestamp(start).date() <= d <= pd.Timestamp(end).date():
            return True
    return False


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["scheduled_utc"].dt.hour
    df["day_of_week"] = df["scheduled_utc"].dt.dayofweek
    df["month"] = df["scheduled_utc"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    # encodage cyclique de l'heure (évite de traiter 23h et 0h comme opposés)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_school_holiday"] = df["scheduled_utc"].apply(_is_school_holiday).astype(int)
    return df


# ------------------------------------------------------------------
# 3. Trafic planifié (safe : basé sur les horaires théoriques de TOUS
#    les vols, connus à l'avance dans les plannings compagnies/aéroports)
# ------------------------------------------------------------------
def add_planned_traffic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_h"] = df["scheduled_utc"].dt.floor("h")
    traffic = (
        df.groupby(["icao", "time_h"]).size().rename("planned_traffic_hour").reset_index()
    )
    df = df.merge(traffic, on=["icao", "time_h"], how="left")
    return df


# ------------------------------------------------------------------
# 4. Historique compagnie (retard moyen glissant, DÉCALÉ de HORIZON_DAYS
#    pour ne jamais regarder dans la fenêtre des 72h avant le vol)
# ------------------------------------------------------------------
def add_airline_rolling_stats(
    df: pd.DataFrame, lookback_days: int = LOOKBACK_DAYS, horizon_days: int = HORIZON_DAYS
) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["scheduled_utc"].dt.date)

    daily = (
        df.groupby(["airline", "date"])["delay_minutes"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values(["airline", "date"])
    )
    daily["airline_avg_delay_30d"] = daily.groupby("airline")["mean"].transform(
        lambda s: s.rolling(lookback_days, min_periods=5).mean()
    )
    # décalage : la stat disponible pour un vol du jour J est celle arrêtée
    # au jour J - horizon_days (simule le fait qu'à J-72h on n'a pas les
    # données des 3 derniers jours)
    daily["airline_avg_delay_30d"] = daily.groupby("airline")["airline_avg_delay_30d"].shift(
        horizon_days
    )

    df = df.merge(
        daily[["airline", "date", "airline_avg_delay_30d"]], on=["airline", "date"], how="left"
    )
    # compagnies trop rares / début de période : on impute par la médiane globale
    df["airline_avg_delay_30d"] = df["airline_avg_delay_30d"].fillna(
        df["airline_avg_delay_30d"].median()
    )
    return df


# ------------------------------------------------------------------
# 5. Historique aéroport (à partir de df_airport_delays), même logique
#    de décalage temporel que pour les compagnies.
# ------------------------------------------------------------------
def add_airport_rolling_stats(
    df: pd.DataFrame,
    df_delays: pd.DataFrame,
    lookback_days: int = LOOKBACK_DAYS,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    d = df_delays.copy()
    d["date"] = pd.to_datetime(pd.to_datetime(d["window_from_utc"]).dt.date)

    # IMPORTANT : dep_median_delay_min / arr_median_delay_min sont vides à 100%
    # dans ce jeu de données (bug côté extraction). Les valeurs utilisables sont
    # dans les colonnes texte dep_median_delay / arr_median_delay, au format
    # timedelta ("-00:02:00"), qu'on convertit ici en minutes.
    d["dep_delay_min"] = pd.to_timedelta(d["dep_median_delay"], errors="coerce").dt.total_seconds() / 60
    d["arr_delay_min"] = pd.to_timedelta(d["arr_median_delay"], errors="coerce").dt.total_seconds() / 60

    daily = (
        d.groupby(["icao", "date"])
        .agg(
            dep_delay_min=("dep_delay_min", "mean"),
            arr_delay_min=("arr_delay_min", "mean"),
            dep_cancel_rate=(
                "dep_num_cancelled",
                lambda s: s.sum() / max(d.loc[s.index, "dep_num_total"].sum(), 1),
            ),
        )
        .reset_index()
        .sort_values(["icao", "date"])
    )

    for col in ["dep_delay_min", "arr_delay_min", "dep_cancel_rate"]:
        roll_col = f"airport_{col}_30d"
        daily[roll_col] = daily.groupby("icao")[col].transform(
            lambda s: s.rolling(lookback_days, min_periods=5).mean()
        )
        daily[roll_col] = daily.groupby("icao")[roll_col].shift(horizon_days)

    keep = ["icao", "date"] + [f"airport_{c}_30d" for c in ["dep_delay_min", "arr_delay_min", "dep_cancel_rate"]]
    df = df.merge(daily[keep], on=["icao", "date"], how="left")
    for c in keep[2:]:
        df[c] = df[c].fillna(df[c].median())
    return df


# ------------------------------------------------------------------
# 6. Climatologie météo (proxy de prévision — voir note en bas de fichier)
# ------------------------------------------------------------------
def build_weather_climatology(df_meteo: pd.DataFrame) -> pd.DataFrame:
    m = df_meteo.copy()
    m["time"] = pd.to_datetime(m["time"])
    m["month"] = m["time"].dt.month
    m["hour"] = m["time"].dt.hour

    agg_cols = ["temperature_2m", "wind_speed_10m", "precipitation", "cloud_cover", "wind_gusts_10m"]
    climo = m.groupby(["icao", "month", "hour"])[agg_cols].mean().reset_index()
    climo = climo.rename(columns={c: f"{c}_climo" for c in agg_cols})
    return climo


def add_weather_climatology(df: pd.DataFrame, climo: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["scheduled_utc"].dt.month
    df["hour"] = df["scheduled_utc"].dt.hour
    df = df.merge(climo, on=["icao", "month", "hour"], how="left")
    return df


# ------------------------------------------------------------------
# 7. Encodages compagnie / destination (fréquence — pas de target encoding
#    pour éviter toute fuite ; simple à recalculer en production)
# ------------------------------------------------------------------
def add_frequency_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["airline_freq"] = df.groupby("airline")["airline"].transform("count")
    df["destination_freq"] = df["destination_icao"].map(
        df["destination_icao"].value_counts()
    ).fillna(0)
    df["is_departure"] = (df["type"] == "departure").astype(int)
    return df


# ------------------------------------------------------------------
# 8. Assemblage complet
# ------------------------------------------------------------------
def build_training_dataset(
    df_mouvs: pd.DataFrame, df_meteo: pd.DataFrame, df_delays: pd.DataFrame
) -> pd.DataFrame:
    df = load_and_clean_mouvs(df_mouvs)
    df = add_calendar_features(df)
    df = add_planned_traffic(df)
    df = add_airline_rolling_stats(df)
    df = add_airport_rolling_stats(df, df_delays)
    climo = build_weather_climatology(df_meteo)
    df = add_weather_climatology(df, climo)
    df = add_frequency_encodings(df)
    return df


# ------------------------------------------------------------------
# 9. Tables de référence pour l'API (calculées une fois/jour, chargées
#    en mémoire au démarrage de l'API — pas de recalcul à chaque requête)
# ------------------------------------------------------------------
def build_reference_tables(
    df_mouvs: pd.DataFrame, df_meteo: pd.DataFrame, df_delays: pd.DataFrame,
    lookback_days: int = LOOKBACK_DAYS,
) -> dict:
    """Construit, à partir des données les plus récentes disponibles, les
    tables d'agrégats que l'API consultera pour chaque prédiction. Pas de
    décalage HORIZON_DAYS ici : au moment du refresh quotidien, on veut la
    stat la plus fraîche possible (le décalage 72h est géré par le fait que
    ce cache n'est lui-même rafraîchi qu'une fois par jour)."""
    df = load_and_clean_mouvs(df_mouvs)
    max_date = df["scheduled_utc"].max()
    recent = df[df["scheduled_utc"] >= max_date - pd.Timedelta(days=lookback_days)]

    airline_stats = (
        recent.groupby("airline")["delay_minutes"].mean().rename("airline_avg_delay_30d").reset_index()
    )
    airline_freq = df["airline"].value_counts().rename("airline_freq").reset_index()
    airline_freq.columns = ["airline", "airline_freq"]
    airline_table = airline_stats.merge(airline_freq, on="airline", how="outer")

    destination_freq = df["destination_icao"].value_counts().rename("destination_freq").reset_index()
    destination_freq.columns = ["destination_icao", "destination_freq"]

    d = df_delays.copy()
    d["date"] = pd.to_datetime(pd.to_datetime(d["window_from_utc"]).dt.date).dt.tz_localize("UTC")
    d["dep_delay_min"] = pd.to_timedelta(d["dep_median_delay"], errors="coerce").dt.total_seconds() / 60
    d["arr_delay_min"] = pd.to_timedelta(d["arr_median_delay"], errors="coerce").dt.total_seconds() / 60
    d_recent = d[d["date"] >= max_date - pd.Timedelta(days=lookback_days)]
    airport_stats = (
        d_recent.groupby("icao")
        .agg(
            airport_dep_delay_min_30d=("dep_delay_min", "mean"),
            airport_arr_delay_min_30d=("arr_delay_min", "mean"),
            airport_dep_cancel_rate_30d=(
                "dep_num_cancelled", lambda s: s.sum() / max(d_recent.loc[s.index, "dep_num_total"].sum(), 1)
            ),
        )
        .reset_index()
    )

    # Trafic planifié "typique" par aéroport / heure / jour de semaine
    # (proxy climatologique : à l'appel de l'API on n'a pas forcément le
    # planning exact des autres vols de l'heure ciblée)
    df["time_h"] = df["scheduled_utc"].dt.floor("h")
    traffic = df.groupby(["icao", "time_h"]).size().rename("count").reset_index()
    traffic["hour"] = traffic["time_h"].dt.hour
    traffic["day_of_week"] = traffic["time_h"].dt.dayofweek
    traffic_climo = (
        traffic.groupby(["icao", "hour", "day_of_week"])["count"]
        .mean().rename("planned_traffic_hour").reset_index()
    )

    weather_climo = build_weather_climatology(df_meteo)

    return {
        "airline_table": airline_table,
        "destination_freq": destination_freq,
        "airport_stats": airport_stats,
        "traffic_climo": traffic_climo,
        "weather_climo": weather_climo,
        "global_medians": {
            "airline_avg_delay_30d": airline_table["airline_avg_delay_30d"].median(),
            "airline_freq": airline_table["airline_freq"].median(),
            "destination_freq": destination_freq["destination_freq"].median(),
            "planned_traffic_hour": traffic_climo["planned_traffic_hour"].median(),
            **{c: airport_stats[c].median() for c in
               ["airport_dep_delay_min_30d", "airport_arr_delay_min_30d", "airport_dep_cancel_rate_30d"]},
        },
        "refreshed_at": pd.Timestamp.utcnow().isoformat(),
    }


def save_reference_tables(tables: dict, output_dir: str) -> None:
    import json
    import os

    os.makedirs(output_dir, exist_ok=True)
    tables["airline_table"].to_parquet(f"{output_dir}/airline_table.parquet")
    tables["destination_freq"].to_parquet(f"{output_dir}/destination_freq.parquet")
    tables["airport_stats"].to_parquet(f"{output_dir}/airport_stats.parquet")
    tables["traffic_climo"].to_parquet(f"{output_dir}/traffic_climo.parquet")
    tables["weather_climo"].to_parquet(f"{output_dir}/weather_climo.parquet")
    with open(f"{output_dir}/meta.json", "w") as f:
        json.dump({"global_medians": tables["global_medians"], "refreshed_at": tables["refreshed_at"]}, f)


def load_reference_tables(input_dir: str) -> dict:
    import json

    with open(f"{input_dir}/meta.json") as f:
        meta = json.load(f)
    return {
        "airline_table": pd.read_parquet(f"{input_dir}/airline_table.parquet"),
        "destination_freq": pd.read_parquet(f"{input_dir}/destination_freq.parquet"),
        "airport_stats": pd.read_parquet(f"{input_dir}/airport_stats.parquet"),
        "traffic_climo": pd.read_parquet(f"{input_dir}/traffic_climo.parquet"),
        "weather_climo": pd.read_parquet(f"{input_dir}/weather_climo.parquet"),
        "global_medians": meta["global_medians"],
        "refreshed_at": meta["refreshed_at"],
    }


def build_features_for_single_flight(
    icao: str, scheduled_utc, airline: str, destination_icao: str | None,
    is_departure: bool, ref: dict,
) -> pd.DataFrame:
    """Construit le vecteur de features pour UN vol, à partir des tables
    de référence mises en cache (rafraîchies 1x/jour). Retourne un DataFrame
    à une ligne, dans l'ordre attendu par le modèle (FEATURE_COLUMNS)."""
    ts = pd.Timestamp(scheduled_utc)
    hour, dow, month = ts.hour, ts.dayofweek, ts.month
    med = ref["global_medians"]

    row = {
        "is_departure": int(is_departure),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_of_week": dow,
        "month": month,
        "is_weekend": int(dow in (5, 6)),
        "is_school_holiday": int(_is_school_holiday(ts)),
    }

    traffic = ref["traffic_climo"].query("icao == @icao and hour == @hour and day_of_week == @dow")
    row["planned_traffic_hour"] = (
        traffic["planned_traffic_hour"].iloc[0] if len(traffic) else med["planned_traffic_hour"]
    )

    al = ref["airline_table"].query("airline == @airline")
    row["airline_avg_delay_30d"] = al["airline_avg_delay_30d"].iloc[0] if len(al) and pd.notna(al["airline_avg_delay_30d"].iloc[0]) else med["airline_avg_delay_30d"]
    row["airline_freq"] = al["airline_freq"].iloc[0] if len(al) and pd.notna(al["airline_freq"].iloc[0]) else med["airline_freq"]

    dest = ref["destination_freq"].query("destination_icao == @destination_icao") if destination_icao else None
    row["destination_freq"] = dest["destination_freq"].iloc[0] if dest is not None and len(dest) else med["destination_freq"]

    ap = ref["airport_stats"].query("icao == @icao")
    for c in ["airport_dep_delay_min_30d", "airport_arr_delay_min_30d", "airport_dep_cancel_rate_30d"]:
        row[c] = ap[c].iloc[0] if len(ap) and pd.notna(ap[c].iloc[0]) else med[c]

    wc = ref["weather_climo"].query("icao == @icao and month == @month and hour == @hour")
    weather_cols = ["temperature_2m_climo", "wind_speed_10m_climo", "precipitation_climo",
                     "cloud_cover_climo", "wind_gusts_10m_climo"]
    if len(wc):
        for c in weather_cols:
            row[c] = wc[c].iloc[0]
    else:
        # secours : moyenne toutes heures/mois confondus pour cet aéroport
        wc_airport = ref["weather_climo"].query("icao == @icao")
        for c in weather_cols:
            row[c] = wc_airport[c].mean() if len(wc_airport) else 0.0

    return pd.DataFrame([row])[FEATURE_COLUMNS]


FEATURE_COLUMNS = [
    "is_departure", "hour_sin", "hour_cos", "day_of_week", "month",
    "is_weekend", "is_school_holiday", "planned_traffic_hour",
    "airline_avg_delay_30d", "airline_freq", "destination_freq",
    "airport_dep_delay_min_30d", "airport_arr_delay_min_30d", "airport_dep_cancel_rate_30d",
    "temperature_2m_climo", "wind_speed_10m_climo", "precipitation_climo",
    "cloud_cover_climo", "wind_gusts_10m_climo",
]
TARGET_COLUMN = "delay_minutes"


if __name__ == "__main__":
    df_mouvs = pd.read_parquet("/mnt/user-data/uploads/df_mouvs_179jours_20260329_224426.parquet")
    df_meteo = pd.read_parquet("/mnt/user-data/uploads/df_meteo_179jours_20260329_224426.parquet")
    df_delays = pd.read_parquet("/mnt/user-data/uploads/df_airport_delays_179jours_20260329_224426.parquet")

    dataset = build_training_dataset(df_mouvs, df_meteo, df_delays)

    print(f"\n[dataset] {dataset.shape[0]:,} lignes x {dataset.shape[1]} colonnes")
    print(f"[dataset] NaN restants sur les features :")
    print(dataset[FEATURE_COLUMNS].isna().sum()[lambda s: s > 0])
    print(f"\n[dataset] Target delay_minutes : min={dataset[TARGET_COLUMN].min()}, "
          f"median={dataset[TARGET_COLUMN].median()}, max={dataset[TARGET_COLUMN].max()}")
