"""
FlyOnTime — API de prédiction du retard (en minutes) des vols commerciaux,
au départ et à l'arrivée, à horizon 72h.

Le modèle et les tables de référence (cache rafraîchi 1x/jour par
refresh_cache.py) sont chargés une fois au démarrage, pas à chaque requête.

Documentation interactive : /docs (Swagger UI) ou /redoc.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import mlflow.pyfunc
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from engine import save_route_prediction_log, fetch_recent_logs
from feature_engineering import build_features_for_single_flight, load_reference_tables, COVERED_AIRPORTS
from flights_provider import get_upcoming_flights
from refresh_cache import refresh as refresh_cache

MODEL_PATH = "model"
CACHE_DIR = "cache"
VALIDATION_SAMPLE_PATH = "validation_sample.parquet"
VALIDATION_META_PATH = "validation_meta.json"

app = FastAPI(
    title="FlyOnTime API",
    description=(
        "Prédit, en minutes, le retard d'un vol commercial au départ ou à "
        "l'arrivée d'un des 5 plus grands aéroports français, jusqu'à 72h "
        "à l'avance. Le modèle (XGBoost) et les agrégats d'historique sont "
        "chargés une fois au démarrage ; le cache d'agrégats est ensuite "
        "rafraîchi automatiquement une fois par jour par une tâche planifiée "
        "intégrée à l'API (voir refresh_cache.py)."
    ),
    version="1.0.0",
    contact={"name": "FlyOnTime — projet de fin de formation"},
)

_state = {"model": None, "ref": None}
_scheduler = BackgroundScheduler()


def _scheduled_refresh():
    """Appelée une fois par jour par le scheduler. Ne bloque jamais l'API :
    en cas d'échec (S3 indisponible, etc.), on garde le cache précédent."""
    try:
        new_ref = refresh_cache(push_to_s3=False)
        _state["ref"] = new_ref
        print(f"[scheduler] Cache rafraîchi automatiquement ({new_ref['refreshed_at']})")
    except Exception as e:
        print(f"[warn] Rafraîchissement planifié échoué, cache précédent conservé : {e}")


@app.on_event("startup")
def load_resources():
    _state["model"] = mlflow.pyfunc.load_model(MODEL_PATH)
    _state["ref"] = load_reference_tables(CACHE_DIR)
    print(f"[startup] Modèle et cache chargés (cache du {_state['ref']['refreshed_at']})")

    try:
        import json
        _state["validation_sample"] = pd.read_parquet(VALIDATION_SAMPLE_PATH)
        with open(VALIDATION_META_PATH) as f:
            _state["validation_meta"] = json.load(f)
        print(f"[startup] Échantillon de validation chargé "
              f"({_state['validation_meta']['n_sample']} vols, MAE {_state['validation_meta']['mae']} min)")
    except FileNotFoundError:
        _state["validation_sample"] = None
        _state["validation_meta"] = None
        print("[warn] Échantillon de validation introuvable — endpoint /model/validation indisponible")

    # Rafraîchissement quotidien intégré — remplace le Cron Job séparé
    # utilisé sur Render ; ici l'API et le job tournent dans le même
    # conteneur (Hugging Face Spaces n'a pas de Cron Job natif).
    _scheduler.add_job(_scheduled_refresh, "interval", hours=24, id="daily_cache_refresh")
    _scheduler.start()


@app.on_event("shutdown")
def stop_scheduler():
    _scheduler.shutdown(wait=False)


class PredictionRequest(BaseModel):
    icao: str = Field(
        ...,
        description=f"Code OACI de l'aéroport. Aéroports couverts : {', '.join(COVERED_AIRPORTS)}.",
        examples=["LFPG"],
    )
    flight_type: str = Field(
        ...,
        description="Type de mouvement : 'departure' (départ) ou 'arrival' (arrivée).",
        examples=["departure"],
    )
    scheduled_utc: datetime = Field(
        ...,
        description="Horaire théorique du vol, en UTC, format ISO 8601. "
                    "Peut être jusqu'à 72h dans le futur.",
        examples=["2026-04-01T08:30:00Z"],
    )
    airline: str = Field(
        ...,
        description="Nom de la compagnie aérienne. Si inconnue du modèle, "
                    "une valeur médiane est utilisée en repli.",
        examples=["Air France"],
    )
    destination_icao: Optional[str] = Field(
        None,
        description="Code OACI de l'aéroport de destination (pertinent pour les "
                    "départs). Optionnel — une valeur médiane est utilisée si absent.",
        examples=["LFML"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "icao": "LFPG",
                "flight_type": "departure",
                "scheduled_utc": "2026-04-01T08:30:00Z",
                "airline": "Air France",
                "destination_icao": "LFML",
            }
        }


class PredictionResponse(BaseModel):
    predicted_delay_minutes: float = Field(..., description="Retard prédit, en minutes (peut être négatif = avance).")
    icao: str
    flight_type: str
    scheduled_utc: datetime
    cache_refreshed_at: str = Field(..., description="Date de dernier rafraîchissement du cache d'agrégats utilisé.")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_delay_minutes": 21.8,
                "icao": "LFPG",
                "flight_type": "departure",
                "scheduled_utc": "2026-04-01T08:30:00Z",
                "cache_refreshed_at": "2026-08-27T08:38:05.338521+00:00",
            }
        }


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' si le modèle et le cache sont chargés, sinon 'not_ready'.")
    cache_refreshed_at: Optional[str] = None


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["monitoring"],
    summary="Vérifie que l'API est prête à servir des prédictions",
)
def health():
    return {
        "status": "ok" if _state["model"] is not None else "not_ready",
        "cache_refreshed_at": _state["ref"]["refreshed_at"] if _state["ref"] else None,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prédiction"],
    summary="Prédit le retard d'un vol",
    responses={
        400: {"description": "Aéroport non couvert par le modèle"},
        422: {"description": "flight_type invalide (doit être 'departure' ou 'arrival')"},
        503: {"description": "Modèle ou cache non chargés (redémarrage en cours)"},
    },
)
def predict(req: PredictionRequest):
    """
    Construit les features du vol à partir des tables de référence mises en
    cache (historique compagnie/aéroport, climatologie météo, trafic
    planifié) et retourne le retard prédit en minutes.

    Chaque prédiction est journalisée dans NeonDB (best-effort : un échec de
    log n'empêche pas la réponse).
    """
    if _state["model"] is None or _state["ref"] is None:
        raise HTTPException(status_code=503, detail="Modèle ou cache non chargés")
    if req.flight_type not in ("departure", "arrival"):
        raise HTTPException(status_code=422, detail="flight_type doit être 'departure' ou 'arrival'")
    if req.icao not in COVERED_AIRPORTS:
        raise HTTPException(
            status_code=400,
            detail=f"Aéroport '{req.icao}' non couvert par le modèle. "
                   f"Aéroports disponibles : {', '.join(COVERED_AIRPORTS)}",
        )

    X = build_features_for_single_flight(
        icao=req.icao,
        scheduled_utc=req.scheduled_utc,
        airline=req.airline,
        destination_icao=req.destination_icao,
        is_departure=(req.flight_type == "departure"),
        ref=_state["ref"],
    )
    predicted = float(_state["model"].predict(X)[0])

    return PredictionResponse(
        predicted_delay_minutes=round(predicted, 1),
        icao=req.icao,
        flight_type=req.flight_type,
        scheduled_utc=req.scheduled_utc,
        cache_refreshed_at=_state["ref"]["refreshed_at"],
    )


class RouteLogRequest(BaseModel):
    icao_dep: str
    icao_arr: str
    airline: str
    scheduled_utc: datetime
    dep_delay_minutes: float
    arr_delay_minutes: float


@app.post(
    "/predictions/log-route",
    tags=["prédiction"],
    summary="Journalise une prédiction de trajet complet (décollage + arrivée) dans NeonDB",
)
def log_route_prediction(req: RouteLogRequest):
    """Appelé par le dashboard une fois les deux prédictions (décollage +
    arrivée) obtenues, pour n'écrire qu'UNE ligne par vol prédit dans
    NeonDB plutôt qu'une ligne par jambe. Best-effort : ne bloque jamais
    l'appelant si NeonDB est indisponible."""
    try:
        save_route_prediction_log(
            req.icao_dep, req.icao_arr, req.airline, req.scheduled_utc,
            req.dep_delay_minutes, req.arr_delay_minutes,
        )
    except Exception as e:
        print(f"[warn] Log NeonDB échoué : {e}")
    return {"status": "logged"}


class RecentPredictionLog(BaseModel):
    timestamp: datetime
    icao_dep: str
    icao_arr: str
    airline: str
    scheduled_utc: Optional[datetime] = None
    dep_delay_minutes: float
    arr_delay_minutes: float


@app.get(
    "/predictions/recent",
    response_model=list[RecentPredictionLog],
    tags=["prédiction"],
    summary="Historique des dernières prédictions journalisées (NeonDB)",
)
def recent_predictions(limit: int = 10):
    """Expose l'historique NeonDB via l'API plutôt que de laisser le
    dashboard se connecter directement à la base — le dashboard n'a ainsi
    aucun secret de base de données à gérer."""
    rows = fetch_recent_logs(limit=limit)
    return [
        RecentPredictionLog(
            timestamp=r[0], icao_dep=r[1], icao_arr=r[2], airline=r[3],
            scheduled_utc=r[4], dep_delay_minutes=r[5], arr_delay_minutes=r[6],
        )
        for r in rows
    ]


class UpcomingFlight(BaseModel):
    flight_number: str
    airline: str
    scheduled_utc: str
    destination_icao: Optional[str] = None
    origin_icao: Optional[str] = None
    label: str


@app.get(
    "/flights/upcoming",
    response_model=list[UpcomingFlight],
    tags=["vols"],
    summary="Vols réels programmés dans les 72h à venir (via AeroDataBox)",
    responses={400: {"description": "Aéroport non couvert par le modèle"}},
)
def upcoming_flights(icao: str, flight_type: str, hours: int = 72):
    """Retourne une liste vide (jamais une erreur) si AeroDataBox n'est pas
    configuré ou indisponible — au dashboard de proposer un repli sur des
    vols réalistes de substitution dans ce cas."""
    if icao not in COVERED_AIRPORTS:
        raise HTTPException(status_code=400, detail=f"Aéroport '{icao}' non couvert par le modèle.")
    if flight_type not in ("departure", "arrival"):
        raise HTTPException(status_code=422, detail="flight_type doit être 'departure' ou 'arrival'")
    hours = min(hours, 72)  # jamais au-delà de l'horizon que le modèle sait gérer
    flights = get_upcoming_flights(icao, flight_type, hours_ahead=hours)
    return flights


class DelayIndexResponse(BaseModel):
    icao: str
    avg_predicted_delay_minutes: float
    level: str
    level_label: str
    n_flights_sampled: int
    source: str


@app.get(
    "/airports/{icao}/delay-index",
    response_model=DelayIndexResponse,
    tags=["vols"],
    summary="Niveau de retard moyen prédit pour l'aéroport sur les prochaines 24h",
    responses={400: {"description": "Aéroport non couvert par le modèle"}},
)
def airport_delay_index(icao: str, hours: int = 24):
    """Fait tourner le modèle sur un échantillon de vols à venir (réels si
    AeroDataBox est configuré, sinon vols réalistes de substitution) et
    moyenne les prédictions pour donner une tendance jour — utile par
    exemple pour anticiper des stocks/effectifs en boutique aéroport.

    Seuils : ≤15 min = pas de retard moyen, 15-30 min = peu de retard
    moyen, >30 min = retard important.
    """
    if icao not in COVERED_AIRPORTS:
        raise HTTPException(status_code=400, detail=f"Aéroport '{icao}' non couvert par le modèle.")
    if _state["model"] is None or _state["ref"] is None:
        raise HTTPException(status_code=503, detail="Modèle ou cache non chargés")

    hours = min(hours, 72)
    source = "aerodatabox"
    sample = []
    for flight_type in ("departure", "arrival"):
        real_flights = get_upcoming_flights(icao, flight_type, hours_ahead=hours)
        for f in real_flights:
            sample.append({
                "scheduled_utc": f["scheduled_utc"],
                "airline": f["airline"],
                "destination_icao": f.get("destination_icao"),
                "is_departure": flight_type == "departure",
            })

    if not sample:
        # Repli : échantillon synthétique régulier sur l'horizon, avec les
        # compagnies les plus fréquentes de l'aéroport (mêmes données que
        # celles utilisées à l'entraînement) plutôt que de renvoyer une
        # erreur.
        source = "estimation (compagnies fréquentes, sans données de vol réelles)"
        now = datetime.now(timezone.utc)
        for h in range(1, hours, 3):
            sample.append({
                "scheduled_utc": (now + timedelta(hours=h)).isoformat(),
                "airline": "Air France",
                "destination_icao": None,
                "is_departure": h % 2 == 0,
            })

    predictions = []
    for f in sample:
        try:
            X = build_features_for_single_flight(
                icao=icao,
                scheduled_utc=f["scheduled_utc"],
                airline=f["airline"],
                destination_icao=f["destination_icao"],
                is_departure=f["is_departure"],
                ref=_state["ref"],
            )
            predictions.append(float(_state["model"].predict(X)[0]))
        except Exception as e:
            print(f"[warn] Prédiction ignorée dans l'échantillon delay-index : {e}")

    if not predictions:
        raise HTTPException(status_code=503, detail="Impossible de calculer un indicateur (aucune prédiction valide)")

    avg_delay = sum(predictions) / len(predictions)
    if avg_delay <= 15:
        level, level_label = "faible", "Pas de retard moyen"
    elif avg_delay <= 30:
        level, level_label = "modere", "Peu de retard moyen"
    else:
        level, level_label = "important", "Retard important"

    return DelayIndexResponse(
        icao=icao,
        avg_predicted_delay_minutes=round(avg_delay, 1),
        level=level,
        level_label=level_label,
        n_flights_sampled=len(predictions),
        source=source,
    )


class ValidationPoint(BaseModel):
    icao: str
    is_departure: bool
    scheduled_utc: str
    airline: str
    actual_delay_minutes: float
    predicted_delay_minutes: float


class ValidationResponse(BaseModel):
    mae: float
    rmse: float
    n_test_total: int
    n_sample: int
    test_period_start: str
    test_period_end: str
    sample: list[ValidationPoint]


@app.get(
    "/model/validation",
    response_model=ValidationResponse,
    tags=["prédiction"],
    summary="Prédictions du modèle comparées à la réalité, sur le jeu de test (vols jamais vus à l'entraînement)",
)
def model_validation():
    """Expose un échantillon du jeu de test temporel (21 derniers jours de
    données, jamais utilisés pour entraîner le modèle) avec, pour chaque
    vol, le retard réellement observé et celui prédit par le modèle. Les
    métriques (MAE, RMSE) sont calculées sur l'intégralité du jeu de test,
    pas seulement sur l'échantillon renvoyé."""
    sample_df = _state.get("validation_sample")
    meta = _state.get("validation_meta")
    if sample_df is None or meta is None:
        raise HTTPException(status_code=503, detail="Échantillon de validation non disponible")

    return ValidationResponse(
        **meta,
        sample=[ValidationPoint(**row) for row in sample_df.to_dict(orient="records")],
    )