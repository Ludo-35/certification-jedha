# Certification Jedha — Data Science

Ce repository regroupe l'ensemble des projets réalisés dans le cadre de ma certification Data Science chez **Jedha Bootcamp**, organisés par bloc de compétences.

## 📋 Vue d'ensemble des blocs

| Bloc | Intitulé | Projet(s) |
|------|----------|-----------|
| **B1** | Construction et alimentation d'une infrastructure de gestion de données | Kayak |
| **B2** | Analyse exploratoire, descriptive et inférentielle de données | Speed Dating (Tinder), Steam (Databricks/Ubisoft) |
| **B3** | Analyse prédictive de données structurées par intelligence artificielle | Walmart, Uber (clustering) |
| **B4** | Analyse prédictive de données non structurées par IA | AT&T (détection de spam) |
| **B5** | Industrialisation d'un algorithme d'apprentissage automatique et automatisation des processus de décision | Getaround (API de pricing) |
| **B6** | Direction de projets de gestion de données (PPML) | FlyOnTime (prédiction de retards de vols) |

---

## B1 — Construction et alimentation d'une infrastructure de gestion de données
**Projet : Kayak**

Mise en place d'un pipeline de collecte et de structuration de données (scraping, API, stockage) pour alimenter une infrastructure de données destinée à l'analyse de destinations de voyage.

📁 [`B1_construction_alimentation_infra_gestion_donnees/`](./B1_construction_alimentation_infra_gestion_donnees)

---

## B2 — Analyse exploratoire, descriptive et inférentielle de données
**Projets : Speed Dating (Tinder) & Steam (Databricks/Ubisoft)**

- **Speed Dating** : analyse statistique descriptive et inférentielle du dataset Speed Dating (2002-2004) dans le cadre d'un cas fictif pour Tinder — corrélations, visualisations et interprétations, restituées lors d'une présentation orale devant jury.
- **Steam** : analyse exploratoire à grande échelle du catalogue de jeux Steam avec PySpark sur Databricks, dans le cadre d'une mission fictive pour Ubisoft (genres, plateformes, prix, impact du COVID).

📁 [`B2_analyse exploratoire_descriptive_inferentielle_donnees_tinder_steam_2_a_deposer_1a_pres_term_saufaws/`](<./B2_analyse exploratoire_descriptive_inferentielle_donnees_tinder_steam_2_a_deposer_1a_pres_term_saufaws>)

---

## B3 — Analyse prédictive de données structurées par intelligence artificielle
**Projets : Walmart & Uber**

- **Walmart** : prédiction des ventes hebdomadaires par régression (linéaire, Ridge, Lasso) à partir de données structurées (EDA, feature engineering, régularisation).
- **Uber** : clustering non supervisé (KMeans, DBSCAN) des prises en charge à New York pour identifier les zones à forte demande, avec tableau de bord de visualisation.

📁 [`B3_analyse_pred_donnees_structurees_par_intelligence_artificielle_walmart_uber_north_deposer3_presenter1/`](<./B3_analyse_pred_donnees_structurees_par_intelligence_artificielle_walmart_uber_north_deposer3_presenter1>)

---

## B4 — Analyse prédictive de données non structurées par IA
**Projet : AT&T**

Détecteur de spam SMS par apprentissage profond, à partir du seul contenu textuel des messages. Plusieurs architectures testées : baseline Embedding+Dense, RNN, transfer learning avec DistilBERT (générique et spécialisé), avec feature engineering complémentaire.

📁 [`B4_analyse_pred_de_donnee_non_structurées_par_ia_ATT_obligatoire/`](<./B4_analyse_pred_de_donnee_non_structurées_par_ia_ATT_obligatoire>)

---

## B5 — Industrialisation d'un algorithme d'apprentissage automatique
**Projet : Getaround**

Système complet de prédiction du prix optimal de location de véhicules : API FastAPI (endpoint `/predict`) servant un modèle scikit-learn, accompagnée d'un dashboard Streamlit d'analyse des retards de restitution. Déployé sur Render (API + dashboard).

📁 [`B5_industrialisation_algorithme_apprentissage_automatique_processus_decision_getaround_obligatoire/`](<./B5_industrialisation_algorithme_apprentissage_automatique_processus_decision_getaround_obligatoire>)

---

## B6 — Direction de projets de gestion de données (PPML)
**Projet : FlyOnTime**

Capstone MLOps : système de prédiction des retards de vols (départ et arrivée) à 72h pour les cinq plus grands aéroports français. Pipeline orchestré avec Airflow, modèle CatBoost suivi via MLflow, API + dashboard conteneurisés avec Docker, stockage S3/NeonDB.

📁 [`B6-direction_de_projets_de_gestion_donnée_PPML_obligatoire/`](<./B6-direction_de_projets_de_gestion_donnée_PPML_obligatoire>)

---

## 🛠️ Stack technique globale

- **Langages** : Python, SQL
- **Data & ML** : Pandas, Scikit-learn, PySpark, CatBoost, XGBoost, DistilBERT
- **MLOps** : MLflow, Airflow, Docker
- **Déploiement** : FastAPI, Streamlit, Render, Hugging Face Spaces
- **Stockage** : AWS S3, NeonDB (PostgreSQL)

---

## 👤 Auteur

**Ludo** — Formation Data Science, Jedha Bootcamp
