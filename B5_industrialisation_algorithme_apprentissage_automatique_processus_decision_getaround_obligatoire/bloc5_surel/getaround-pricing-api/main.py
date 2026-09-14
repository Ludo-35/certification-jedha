import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Any

app = FastAPI(
    title="Getaround Pricing API",
    description="API de prédiction de prix optimaux pour la location de véhicules Getaround.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Le fichier model.joblib n'existe pas. Veuillez exécuter train_model.py d'abord.")

model = joblib.load(MODEL_PATH)

class PredictionInput(BaseModel):
    input: List[List[Any]]

FEATURE_NAMES = [
    'model_key', 'mileage', 'engine_power', 'fuel', 'paint_color',
    'car_type', 'private_parking_available', 'has_gps',
    'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
    'has_speed_regulator', 'winter_tires'
]

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API Getaround. Rendez-vous sur /docs pour la documentation."}

@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        data = payload.input
        if len(data[0]) == len(FEATURE_NAMES):
            df_input = pd.DataFrame(data, columns=FEATURE_NAMES)
            predictions = model.predict(df_input)
        else:
            predictions = model.predict(pd.DataFrame(data))
            
        return {"prediction": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))