# FlyOnTime — Prédiction de retards de vols

**Bloc de certification :** B6 — Direction de projets de gestion de données (PPML)
**Auteurs :** Ludo · Patrick — Jedha Bootcamp

## Objectif

Système de prédiction des retards de vols pour les **cinq plus grands aéroports
français** (Paris CDG, Paris Orly, Lyon, Nice, Marseille). L'objectif : fournir,
jusqu'à **72h à l'avance**, une estimation en minutes des retards au décollage
ET à l'arrivée pour un vol commercial donné, via un dashboard interactif.

Cas d'usage concret ayant guidé les choix d'UX : une boutique en zone
aéroportuaire qui doit anticiper ses stocks et ses effectifs selon les
retards attendus.

## Approche & choix de modélisation

- **Modèle unique** : XGBoost, tracké via MLflow. Cible `delay_minutes`, avec
  un flag `is_departure` distinguant décollage/arrivée plutôt que deux
  modèles séparés — plus de volume d'entraînement, motifs communs (météo,
  trafic) mutualisés.
- **Fuite de données évitée** : seule `scheduled_utc` (l'horaire théorique)
  est utilisable comme feature temporelle. `revised_utc`, `runway_utc` et
  `status` sont exclus, car indisponibles au moment réel de la prédiction
  (J-72h).
- **Split temporel** (pas aléatoire) : entraînement jusqu'à début mars 2026,
  test sur les 3 dernières semaines, avec un **gap de sécurité de 3 jours**
  entre les deux — les features d'historique glissant ne regardent jamais
  dans la fenêtre des 72h précédant un vol du jeu de test.
- **Météo** : climatologie (moyenne historique par aéroport/mois/heure), et
  non une vraie prévision météo — faute d'avoir branché une API de prévision
  à J-72h. C'est une limitation assumée et documentée, pas une prévision
  réelle.
- **Features d'historique** (décalées de 3 jours) : retard moyen glissant
  30j par compagnie et par aéroport, taux d'annulation, densité de trafic
  planifié, calendrier (heure, jour, vacances scolaires françaises).
- **Résultats** : MAE 12,75 min / RMSE 22,96 min sur 111 410 vols de test —
  meilleur que toutes les baselines testées (historique compagnie seul :
  14,03 min ; moyenne globale : 17,29 min).

## Architecture

```
data → training/ (entraîne le modèle) → model/
                                            │
                     api/ (FastAPI) ───────┘  ← charge modèle + cache une fois au démarrage
                        │  /predict, /flights/upcoming, /airports/{icao}/delay-index,
                        │  /model/validation, /predictions/recent, /predictions/log-route
                        │  rafraîchit son cache d'historique 1x/jour (tâche planifiée intégrée)
                        │
                dashboard/ (Streamlit) → appelle l'API, aucune logique ML/DB
```

- **API FastAPI** : contrat clair, modèle et agrégats d'historique chargés
  une seule fois au démarrage. Interroge **AeroDataBox** (RapidAPI) pour
  proposer de vrais vols programmés, avec repli automatique sur des
  combinaisons compagnie/route réalistes (dérivées des données
  d'entraînement) si l'API externe est indisponible ou rate-limitée.
- **Dashboard Streamlit** séparé, consommant uniquement l'API — deux
  onglets : **Prédiction** (parcours séquentiel décollage → arrivée → date
  → compagnie → horaire, deux indicateurs de retard départ/arrivée, carte
  animée du trajet) et **Validation du modèle** (nuage de points prédit vs
  réel sur le jeu de test, histogramme des erreurs).
- Chaque prédiction de trajet complet (décollage + arrivée) est journalisée
  dans **NeonDB/PostgreSQL** en une seule ligne.

## Pipeline MLOps

- **Modèle** : XGBoost, tracké via MLflow.
- **Stockage** : AWS S3 (données brutes et artefacts), NeonDB/PostgreSQL
  (journal des prédictions).
- **Rafraîchissement du cache** : intégré à l'API via une tâche planifiée
  (APScheduler, 1x/24h) plutôt qu'un Cron Job externe — plus simple dès lors
  que l'API et le rafraîchissement tournent dans le même conteneur.
- **Conteneurisation** : Docker, orchestré via Docker Compose pour la démo.

## Structure du repo

| Dossier | Contenu |
|---|---|
| `api/` | Service FastAPI, intégration AeroDataBox, cache de features |
| `dashboard/` | App Streamlit (Prédiction + Validation du modèle) |
| `training/` | Script d'entraînement XGBoost/MLflow (exécution ponctuelle) |
| `model/` | Modèle entraîné (format MLflow) |
| `cache/` | Tables de référence pré-calculées (historique, climatologie) |
| `data/` | Jeux de données bruts (mouvements, météo, retards agrégés — 179 jours) |
| `legacy/` | Anciens fichiers remplacés en cours de projet, conservés pour traçabilité |
| `scripts/` | Utilitaires de dev (vérification S3, debug colonnes) |

## Démarrage — démo locale via Docker Compose

Les hébergeurs cloud gratuits testés (Render, Hugging Face Spaces, Railway)
exigent désormais tous une carte bancaire enregistrée, même sur leur tier
gratuit. Pour une démo fiable, le projet tourne **en local**.

**Prérequis :** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

1. Copier `.env.example` en `.env` à la racine et renseigner les vraies
   valeurs (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DATABASE_URL`,
   `AERODATABOX_API_KEY`)
2. Depuis la racine du repo :
   ```bash
   docker compose up --build
   ```
3. Ouvrir :
   - **Dashboard** → http://localhost:8501
   - **API** → http://localhost:8000 (doc interactive sur `/docs`)

Le mapping de ports est géré par `docker-compose.yml` — les conteneurs
écoutent en interne sur le port 7860 (convention héritée d'un essai de
déploiement Hugging Face Spaces), mappé sur 8000/8501 côté hôte.

### Base de données — schéma NeonDB requis

Avant le premier lancement, créer la table dans NeonDB (SQL Editor sur
[console.neon.tech](https://console.neon.tech)) :

```sql
CREATE TABLE prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT now(),
    icao_dep TEXT NOT NULL,
    icao_arr TEXT NOT NULL,
    airline TEXT,
    scheduled_utc TIMESTAMPTZ,
    dep_delay_minutes REAL,
    arr_delay_minutes REAL
);
```

### Démarrage sans Docker (dev)

```bash
# API
cd api && pip install -r requirements-api.txt && uvicorn api:app --reload

# Dashboard (autre terminal, API démarrée)
cd dashboard && pip install -r requirements-dashboard.txt
export FLYONTIME_API_URL=http://localhost:8000
streamlit run dashboard.py

# Réentraîner le modèle
cd training && pip install -r requirements-training.txt && python train.py

# Rafraîchir le cache manuellement
cd api && python refresh_cache.py
```

## Déploiement cloud (tenté, non finalisé)

`render.yaml` reste dans le repo comme trace d'un plan de déploiement Render
(API + dashboard + cron) abandonné faute de carte bancaire validée sur le
compte. Une piste Hugging Face Spaces a été tentée en repli, également
bloquée (le SDK Docker y est désormais réservé au tier payant). Ces deux
options restent réutilisables telles quelles si la contrainte est levée.

## Limitations connues

- **Météo climatologique**, pas une vraie prévision à J-72h.
- **Heure d'arrivée approximée** sur l'heure de décollage (même horaire
  utilisé pour les deux prédictions) — les données ne fournissant que
  l'heure de décollage, c'est une approximation raisonnable sur les
  liaisons courtes entre ces 5 aéroports.
- **Rate-limit AeroDataBox** sur le plan RapidAPI gratuit — mitigé par un
  espacement des requêtes et des relances automatiques sur 429, mais peut
  encore se produire ponctuellement (repli automatique et transparent sur
  des vols réalistes dans ce cas).
- `feature_engineering.py` est dupliqué entre `api/` et `training/` (deux
  contextes d'exécution distincts) — à garder synchronisé manuellement.

## Phase de faisabilité (historique)

Avant l'architecture actuelle, une phase d'exploration a permis d'analyser
les trois jeux de données sources (mouvements de vols, météo, indices de
retard agrégés) et de prototyper un premier dashboard intégrant
l'API Open-Meteo et une visualisation radar ADS-B via OpenSky Network.

## Contributions

Projet réalisé avec **Patrick**, qui a développé l'intégration base de
données NeonDB/PostgreSQL initiale (`engine.py`) — débogage conjoint des
problèmes de chaîne de connexion.
