# FlyOnTime

Projet de fin de formation — prévision du retard (en minutes) des vols
commerciaux au départ et à l'arrivée des 5 plus grands aéroports français,
jusqu'à 72h à l'avance.

Ludo · Jedha Bootcamp · Bloc 6 (MLOps)

## Architecture

```
data → training/ (entraîne le modèle) → model/
                                            │
                     api/ (FastAPI) ───────┘  ← charge modèle + cache une fois au démarrage
                        │  sert /predict, /health, /predictions/recent
                        │  rafraîchit le cache automatiquement 1x/jour
                        │  (tâche planifiée intégrée, voir refresh_cache.py)
                        │
                dashboard/ (Streamlit) → appelle l'API, aucune logique ML/DB
```

## Structure du repo

| Dossier | Contenu |
|---|---|
| `api/` | Service FastAPI de prédiction |
| `dashboard/` | App Streamlit consommant l'API |
| `training/` | Script d'entraînement du modèle (exécution ponctuelle) |
| `model/` | Modèle XGBoost entraîné (format MLflow) |
| `cache/` | Tables de référence pré-calculées (historique, climatologie météo) |
| `data/` | Jeux de données bruts (mouvements, météo, retards agrégés — 179 jours) |
| `legacy/` | Anciens fichiers remplacés au cours du projet, conservés pour traçabilité |
| `scripts/` | Utilitaires de dev (vérif S3, debug colonnes) |

## Démo — en local via Docker Compose (méthode recommandée)

Les hébergeurs cloud gratuits testés (Render, Hugging Face Spaces, Railway)
exigent tous désormais une carte bancaire enregistrée, même sur leurs tiers
gratuits. Pour une démo fiable (soutenance, revue), le plus simple et le
plus robuste reste de lancer le projet en local.

**Prérequis :** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

1. Copier `.env.example` en `.env` à la racine et renseigner les vraies
   valeurs (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DATABASE_URL`)
2. Depuis la racine du repo :
   ```bash
   docker compose up --build
   ```
3. Dashboard : http://localhost:8501
   API : http://localhost:8000 (doc interactive sur `/docs`)

## Démarrage local sans Docker (dev)

**API :**
```bash
cd api
pip install -r requirements-api.txt
uvicorn api:app --reload
```

**Dashboard** (dans un autre terminal, l'API doit tourner) :
```bash
cd dashboard
pip install -r requirements-dashboard.txt
export FLYONTIME_API_URL=http://localhost:8000
streamlit run dashboard.py
```

**Réentraîner le modèle :**
```bash
cd training
pip install -r requirements-training.txt
python train.py
```

**Rafraîchir le cache manuellement :**
```bash
cd api
python refresh_cache.py
```

## Déploiement cloud (optionnel, non finalisé)

`render.yaml` reste dans le repo comme trace d'un plan de déploiement
Render (2 web services + 1 cron job) abandonné faute de carte bancaire
validée. Il peut resservir tel quel si cette contrainte est levée plus
tard. Idem pour une piste Hugging Face Spaces (SDK Docker), également
bloquée par une exigence de carte bancaire au moment du projet.

## Points d'architecture à retenir

- **Pas de fuite de données** : seules les features connues à J-72h sont
  utilisées (aucune donnée post-décollage/atterrissage). Voir les
  commentaires dans `feature_engineering.py`.
- **Split temporel** (pas aléatoire) pour l'évaluation, avec un gap de
  sécurité de 3 jours entre train et test — voir `training/train.py`.
- **Rafraîchissement du cache intégré à l'API** (tâche planifiée toutes les
  24h via APScheduler) plutôt qu'un cron externe — plus simple dès lors
  que l'API et le rafraîchissement tournent dans le même conteneur.
- `feature_engineering.py` est dupliqué entre `api/` et `training/` (deux
  contextes d'exécution distincts) — à garder synchronisé manuellement, ou
  à migrer vers un package partagé si le projet grandit.
