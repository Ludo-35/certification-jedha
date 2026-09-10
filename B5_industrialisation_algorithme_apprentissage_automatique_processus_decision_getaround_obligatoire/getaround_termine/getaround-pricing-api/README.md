# Getaround Pricing API

API FastAPI de prédiction de prix optimal pour la location de véhicules Getaround.

## Endpoints

- `POST /predict` : prend en entrée un JSON `{"input": [[...]]}` (features du véhicule) et renvoie `{"prediction": [...]}` (prix prédit par jour).
- `GET /docs` : documentation Swagger auto-générée.

## Installation locale

```bash
pip install -r requirements.txt
python train_model.py   # entraîne et sauvegarde model.joblib
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Déploiement

Déployé sur Render (Docker, service gratuit).
