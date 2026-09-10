# ✈️ FlyOnTime — Prédiction de retards de vols

**Bloc de certification :** B6 — Direction de projets de gestion de données (PPML)

## 🎯 Objectif

Système de prédiction des retards de vols pour les **cinq plus grands aéroports français**. L'objectif est de fournir, jusqu'à **72h en avance**, une estimation en minutes des retards au départ et à l'arrivée pour les vols commerciaux, via un dashboard de visualisation.

## 🧩 Approche & choix de modélisation

- **Cible unique** : `delay_minutes`, avec un flag `type` distinguant départ/arrivée, plutôt que deux modèles séparés.
- **Fuite de données évitée** : seule la variable `scheduled_utc` est utilisable comme feature temporelle — `revised_utc`, `runway_utc` et `status` sont exclus car ils ne seraient pas disponibles au moment de la prédiction réelle.
- **Météo** : utilisation des prévisions météo (et non des données observées `df_meteo`, qui constitueraient une fuite), avec la climatologie comme proxy en complément.
- **Features d'historique glissant** : moyennes de retard par compagnie/aéroport, taux d'annulation, densité de trafic planifié, calendrier et vacances scolaires.

## 🏗️ Architecture

- **API FastAPI** de prédiction : contrat clair, features recalculées à la volée.
- **Dashboard Streamlit** séparé, consommant l'API (pattern déjà validé sur le projet Getaround).
- Les agrégats d'historique (compagnie/aéroport) utilisés par l'API sont **rafraîchis une fois par jour** (cache) plutôt que recalculés à chaque requête.

## ⚙️ Pipeline MLOps

- **Orchestration** : DAG Airflow (environnement local sur WSL2, Ubuntu, Python 3.12, Airflow 2.9.1) pour un **réentraînement bi-hebdomadaire**.
- **Modèle** : CatBoost, tracké via **MLflow**.
- **Stockage** : AWS S3 (données et artefacts), NeonDB/PostgreSQL (intégration base de données).
- **Conteneurisation** : Docker.

## 📁 Structure du code

- `extraction.py`, `transform.py`, `load.py`, `modele.py` — pipeline modulaire.
- `train.py` — entraînement CatBoost/MLflow.
- `app.py` / `appstlite.py` — applications Streamlit (turnaround-time et dashboard léger).
- `engine.py` — helpers S3/NeonDB.
- Notebooks restructurés : `ppml_clean.ipynb`, `humanflyontime.ipynb`.

## 👥 Contributions

Projet réalisé en équipe avec **Patrick**, qui a développé l'intégration base de données NeonDB/PostgreSQL. Débogage conjoint des problèmes de chaîne de connexion (caractères spéciaux, erreurs en lecture seule sur l'endpoint pooler).

## 📊 Phase de faisabilité (historique)

Avant l'architecture finale, une phase d'exploration a permis de :
- Analyser trois datasets sources (mouvements de vols, météo, indices de retard agrégés).
- Développer des scripts de feature engineering (`feature_engineering_mouvs.py`, `merge_datasets.py`) avec lag features, encodage cyclique et target encoding bayésien.
- Prototyper un dashboard HTML/JS avec intégration API Open-Meteo et visualisation radar ADS-B (OpenSky Network).
