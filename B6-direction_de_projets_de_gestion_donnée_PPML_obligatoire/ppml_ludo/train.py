"""
FlyOnTime — Entraînement du modèle de prédiction du retard (en minutes),
départ + arrivée, à horizon 72h.

Point clé : split TEMPOREL, pas aléatoire. Un split aléatoire mélangerait
des vols passés et futurs entre train et test, ce qui est irréaliste
(en production, on prédit toujours le futur à partir du passé) et
optimiste de façon artificielle à cause des features d'historique glissant.
"""

import shutil
import os

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from feature_engineering import build_training_dataset, FEATURE_COLUMNS, TARGET_COLUMN

TEST_DAYS = 21   # ~3 semaines de test, dernière période chronologique
GAP_DAYS = 3     # marge de sécurité = l'horizon de prédiction (72h)


# ------------------------------------------------------------------
# Split temporel avec gap de sécurité
# ------------------------------------------------------------------
def temporal_train_test_split(df: pd.DataFrame, test_days: int = TEST_DAYS, gap_days: int = GAP_DAYS):
    """
    train : tout ce qui précède (max_date - test_days - gap_days)
    gap   : zone retirée entre train et test (évite toute contamination
            via les features d'historique glissant, qui regardent jusqu'à
            gap_days en arrière)
    test  : les test_days derniers jours du dataset
    """
    df = df.sort_values("scheduled_utc")
    max_date = df["scheduled_utc"].max()
    test_start = max_date - pd.Timedelta(days=test_days)
    gap_start = test_start - pd.Timedelta(days=gap_days)

    train = df[df["scheduled_utc"] < gap_start]
    test = df[df["scheduled_utc"] >= test_start]

    print(f"[split] train : {train['scheduled_utc'].min().date()} -> "
          f"{train['scheduled_utc'].max().date()}  ({len(train):,} vols)")
    print(f"[split] gap   : {gap_days} jours retirés (simule l'horizon 72h)")
    print(f"[split] test  : {test['scheduled_utc'].min().date()} -> "
          f"{test['scheduled_utc'].max().date()}  ({len(test):,} vols)")
    return train, test


# ------------------------------------------------------------------
# Baselines de comparaison (pour juger si le modèle apporte vraiment
# quelque chose au-delà de règles simples)
# ------------------------------------------------------------------
def compute_baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    y_test = test[TARGET_COLUMN]

    naive_zero = pd.Series(0, index=test.index)
    naive_global_mean = pd.Series(train[TARGET_COLUMN].mean(), index=test.index)
    naive_airline_hist = test["airline_avg_delay_30d"]  # déjà calculée en feature, sans fuite

    return {
        "baseline_zero (aucun retard prévu)": mean_absolute_error(y_test, naive_zero),
        "baseline_moyenne_globale_train": mean_absolute_error(y_test, naive_global_mean),
        "baseline_historique_compagnie": mean_absolute_error(y_test, naive_airline_hist),
    }


# ------------------------------------------------------------------
# Entraînement
# ------------------------------------------------------------------
def run_full_training():
    print("🚀 Démarrage du pipeline d'entraînement (split temporel)...")

    df_mouvs = pd.read_parquet("/mnt/user-data/uploads/df_mouvs_179jours_20260329_224426.parquet")
    df_meteo = pd.read_parquet("/mnt/user-data/uploads/df_meteo_179jours_20260329_224426.parquet")
    df_delays = pd.read_parquet("/mnt/user-data/uploads/df_airport_delays_179jours_20260329_224426.parquet")

    dataset = build_training_dataset(df_mouvs, df_meteo, df_delays)
    train, test = temporal_train_test_split(dataset)

    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]

    baselines = compute_baselines(train, test)

    mlflow.set_experiment("FlyOnTime_Delay_Prediction")
    with mlflow.start_run(run_name="XGBoost_per_flight_v1"):
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        for name, val in baselines.items():
            mlflow.log_metric(f"baseline_{name.split(' ')[0]}", val)

        print(f"\n📈 Modèle XGBoost :  MAE = {mae:.2f} min   |   RMSE = {rmse:.2f} min")
        print("📊 Baselines (MAE) :")
        for name, val in baselines.items():
            print(f"   - {name:45s} : {val:.2f} min")

        # Breakdown départ / arrivée
        for label, mask in [("departure", X_test["is_departure"] == 1), ("arrival", X_test["is_departure"] == 0)]:
            mae_sub = mean_absolute_error(y_test[mask], preds[mask])
            print(f"   -> MAE {label:10s} : {mae_sub:.2f} min  ({mask.sum():,} vols)")

        # Importance des features
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
        print("\n🔎 Top 10 features (importance XGBoost) :")
        print(importances.head(10).to_string())

        model_path = "model"
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        mlflow.xgboost.save_model(model, model_path, model_format="json")
        print(f"\n📁 Modèle sauvegardé dans '{model_path}/' (format JSON)")

    return model, mae, rmse, baselines


if __name__ == "__main__":
    run_full_training()
