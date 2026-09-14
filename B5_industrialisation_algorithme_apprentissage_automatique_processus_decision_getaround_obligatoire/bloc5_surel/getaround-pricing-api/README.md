# Getaround Pricing API

API FastAPI de prédiction de prix optimal pour la location de véhicules Getaround.

## Endpoints

- `POST /predict` : prend en entrée un JSON `{"input": [[...]]}` (features du véhicule) et renvoie `{"prediction": [...]}` (prix prédit par jour).
- `GET /docs` : documentation Swagger auto-générée.

## Mode opératoire — Accéder à l'API

1. L'API est en ligne à l'adresse : **https://getaround-pricing-api.onrender.com**
2. Service hébergé sur le tier gratuit de Render : après une période d'inactivité, la première requête peut prendre **30 à 60 secondes** avant de répondre (réveil de l'instance) — c'est normal, il suffit de patienter.
3. **Documentation interactive (Swagger)** — la manière la plus simple de tester l'API sans écrire de code :
   - Ouvrir https://getaround-pricing-api.onrender.com/docs
   - Cliquer sur `POST /predict`, puis sur **"Try it out"**
   - Renseigner un JSON d'exemple dans le corps de la requête, par exemple :
     ```json
     {
       "input": [["Citroën", 140411, 100, "diesel", "black", "convertible", true, true, true, true, true, true, true]]
     }
     ```
   - Cliquer sur **"Execute"** : la réponse (prix prédit) s'affiche directement sous la requête.
4. **En ligne de commande (curl)** :
   ```bash
   curl -i -H "Content-Type: application/json" -X POST \
     -d '{"input": [["Citroën", 140411, 100, "diesel", "black", "convertible", true, true, true, true, true, true, true]]}' \
     https://getaround-pricing-api.onrender.com/predict
   ```
5. **En Python** :
   ```python
   import requests

   response = requests.post(
       "https://getaround-pricing-api.onrender.com/predict",
       json={"input": [["Citroën", 140411, 100, "diesel", "black", "convertible", True, True, True, True, True, True, True]]}
   )
   print(response.json())
   ```
6. Réponse attendue dans tous les cas :
   ```json
   {"prediction": [103.45]}
   ```

> ℹ️ L'ordre des valeurs dans `input` doit correspondre à : `model_key, mileage, engine_power, fuel, paint_color, car_type, private_parking_available, has_gps, has_air_conditioning, automatic_car, has_getaround_connect, has_speed_regulator, winter_tires`.

## Installation locale

```bash
pip install -r requirements.txt
python train_model.py   # entraîne et sauvegarde model.joblib
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Déploiement

Déployé sur Render (Docker, service gratuit).