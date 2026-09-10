# Getaround — Projet de certification

Étude de cas Getaround : analyse de l'impact d'un délai minimum entre deux
locations, et API de prédiction du prix optimal de location.

## Livrables en ligne

| Livrable | Lien |
|---|---|
| **Tableau de bord** (analyse des retards) | https://getaround-delay-dashboard.onrender.com |
| **API de prédiction de prix** | https://getaround-pricing-api.onrender.com |
| Documentation de l'API (Swagger) | https://getaround-pricing-api.onrender.com/docs |

## Dépôts GitHub

| Composant | Dépôt |
|---|---|
| API | https://github.com/Ludo-35/getaround-pricing-api |
| Dashboard | https://github.com/Ludo-35/getaround-delay-dashboard |

>  Les deux services tournent sur le service tiers gratuit de Render : ils se
> mettent en veille après 15 min d'inactivité. Le premier appel après une
> période d'inactivité peut donc prendre 30 à 60 secondes.

## Structure du projet

```
ludo_projet/
├── api/            # API FastAPI (prédiction de prix) — déployée sur Render
│   ├── main.py
│   ├── train_model.py
│   ├── model.joblib
│   ├── get_around_pricing_project.csv
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
│
└── dashboard/       # Dashboard Streamlit (analyse des retards) — déployé sur Render
    ├── app.py
    ├── get_around_delay_analysis.xlsx
    ├── requirements.txt
    └── README.md
```

## Tester l'API en local

```bash
cd api
pip install -r requirements.txt
python train_model.py
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Lancer le dashboard en local

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Exemple d'appel à l'API

```bash
curl -i -H "Content-Type: application/json" -X POST \
  -d '{"input": [["Citroën", 140411, 100, "diesel", "black", "convertible", true, true, true, true, true, true, true]]}' \
  https://getaround-pricing-api.onrender.com/predict
```

Réponse attendue :

```json
{"prediction": [103.45]}
```
