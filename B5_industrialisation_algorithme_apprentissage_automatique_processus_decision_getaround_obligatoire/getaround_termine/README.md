# 🚗 Getaround — Analyse des retards & API de pricing

**Bloc de certification :** B5 — Industrialisation d'un algorithme d'apprentissage automatique et automatisation des processus de décision

## 🎯 Contexte

Pour **Getaround** (plateforme de location de voitures entre particuliers), deux problématiques métier :
1. Comprendre l'impact des **retards de restitution** des véhicules sur les locations suivantes.
2. Fournir aux propriétaires un outil de **prédiction du prix optimal** de location.

## 🎯 Objectif

Livrer une solution complète, **industrialisée et déployée** :
- Un **dashboard** d'analyse des retards de restitution
- Une **API de prédiction de prix** exposant un modèle entraîné, prête à être consommée par une application tierce

## 🏗️ Architecture

- **Dashboard** (Streamlit) : analyse exploratoire des retards de restitution, impact sur les réservations suivantes.
- **API** (FastAPI) : endpoint `/predict` qui retourne un prix de location optimal à partir des caractéristiques du véhicule, en s'appuyant sur un modèle **scikit-learn** sérialisé (`model.joblib`, entraîné via `train_model.py`).
- Projet organisé en deux sous-dossiers distincts : `api/` et `dashboard/`.
- **Conteneurisation** Docker pour les deux composants.

## 🚀 Déploiement

- **API** : déployée sur [Render](https://getaround-pricing-api.onrender.com) — endpoint `/predict` fonctionnel, documentation interactive disponible sur `/docs`.
- **Dashboard** : déployé sur [Render](https://getaround-delay-dashboard.onrender.com).

*(Le déploiement initial visait Hugging Face Spaces, qui bloquait le SDK Docker et le hardware CPU basic sur le compte gratuit — bascule effectuée vers Render.)*

## 🛠️ Stack technique

- **API** : FastAPI, Uvicorn
- **Modélisation** : Scikit-learn, Joblib
- **Dashboard** : Streamlit
- **Déploiement** : Docker, Render

## 📬 Livrables

- Dashboard d'analyse des retards, déployé et accessible en ligne
- API de prédiction de prix, déployée et documentée (`/docs`)
