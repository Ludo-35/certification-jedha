"""
FlyOnTime — API de prédiction du retard (en minutes) des vols commerciaux,
au départ et à l'arrivée, à horizon 72h.

Le modèle et les tables de référence (cache rafraîchi 1x/jour par
refresh_cache.py) sont chargés une fois au démarrage, pas à chaque requête.
"""

from datetime import datetime
from typing import Optional

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from engine import save_prediction_log
from feature_engineering import build_features_for_single_flight, load_reference_tables, COVERED_AIRPORTS

MODEL_PATH = "model"
CACHE_DIR = "cache"

app = FastAPI(title="FlyOnTime API", description="Prédiction de retard de vol à J-72h")

_state = {"model": None, "ref": None}


@app.on_event("startup")
def load_resources():
    _state["model"] = mlflow.pyfunc.load_model(MODEL_PATH)
    _state["ref"] = load_reference_tables(CACHE_DIR)
    print(f"[startup] Modèle et cache chargés (cache du {_state['ref']['refreshed_at']})")


class PredictionRequest(BaseModel):
    icao: str = Field(..., description="Code OACI de l'aéroport, ex. 'LFPG'")
    flight_type: str = Field(..., description="'departure' ou 'arrival'")
    scheduled_utc: datetime = Field(..., description="Horaire théorique du vol, UTC, ISO 8601")
    airline: str = Field(..., description="Nom de la compagnie")
    destination_icao: Optional[str] = Field(None, description="Aéroport de destination (départs uniquement)")


class PredictionResponse(BaseModel):
    predicted_delay_minutes: float
    icao: str
    flight_type: str
    scheduled_utc: datetime
    cache_refreshed_at: str


@app.get("/health")
def health():
    return {
        "status": "ok" if _state["model"] is not None else "not_ready",
        "cache_refreshed_at": _state["ref"]["refreshed_at"] if _state["ref"] else None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
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

    # Log asynchrone-friendly : on ne bloque pas la réponse si NeonDB est lent/down
    try:
        save_prediction_log(req.icao, predicted, f"API predict ({req.flight_type})", "n/a")
    except Exception as e:
        print(f"[warn] Log NeonDB échoué : {e}")

    return PredictionResponse(
        predicted_delay_minutes=round(predicted, 1),
        icao=req.icao,
        flight_type=req.flight_type,
        scheduled_utc=req.scheduled_utc,
        cache_refreshed_at=_state["ref"]["refreshed_at"],
    )
