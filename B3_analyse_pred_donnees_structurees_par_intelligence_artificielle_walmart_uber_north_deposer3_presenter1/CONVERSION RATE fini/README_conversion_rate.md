# 📈 Conversion Rate Challenge — Data Science Weekly

**Bloc de certification :** B3 — Analyse prédictive de données structurées par intelligence artificielle

## 🎯 Contexte

**Data Science Weekly**, newsletter spécialisée en data science, souhaite comprendre ce qui pousse un visiteur de son site à s'abonner ("conversion"), afin d'identifier des leviers d'action pour améliorer son taux de conversion.

Le projet est structuré comme une **compétition de type Kaggle** : un fichier `data_train.csv` étiqueté pour l'entraînement, un fichier `data_test.csv` non étiqueté pour la soumission des prédictions, évaluées via le **F1-score** (métrique adaptée à un problème fortement déséquilibré).

## 📊 Données

- **284 580 sessions** utilisateur (jeu d'entraînement)
- 5 variables explicatives : pays, âge, statut nouvel utilisateur, source d'acquisition, nombre de pages visitées
- Cible binaire `converted`, fortement déséquilibrée : **seulement 3,23 % de conversions**

## 🔍 Analyse exploratoire — principaux constats

- **Engagement sur site** : les visiteurs convertis consultent en moyenne 14,6 pages contre 4,6 pour les non-convertis — le signal le plus discriminant.
- **Statut utilisateur** : les utilisateurs existants convertissent 5x mieux (7,2 %) que les nouveaux visiteurs (1,4 %).
- **Effet pays marqué** : Allemagne (6,2 %), UK (5,2 %), US (3,8 %) vs Chine (0,13 % malgré un volume de trafic important).
- **Source d'acquisition** : effet plus modéré (Ads 3,5 %, Seo 3,3 %, Direct 2,8 %).
- 2 valeurs aberrantes d'âge (111 et 123 ans) retirées du jeu d'entraînement.

![Vue d'ensemble EDA](./eda_overview.png)

## 🛠️ Méthodologie de modélisation

- **Prétraitement** : one-hot encoding des variables catégorielles (pays, source), standardisation des variables numériques, split stratifié 80/20.
- **Modèles comparés** : régression logistique (baseline), Random Forest, XGBoost.
- **Gestion du déséquilibre des classes** : une première approche par pondération automatique (`class_weight='balanced'`, `scale_pos_weight`) a donné des résultats décevants (F1 ≈ 0,51–0,57). La stratégie retenue a été d'entraîner sans rééquilibrage forcé puis d'**optimiser a posteriori le seuil de décision** sur la courbe précision-rappel — ce qui a permis de faire passer le F1 de 0,57 à 0,77.
- **Optimisation finale** : recherche sur grille des hyperparamètres XGBoost (profondeur, nombre d'estimateurs, taux d'apprentissage).

## 📈 Résultats

**Modèle final : XGBoost** (profondeur max 4, 200 arbres, learning rate 0,05, seuil de décision 0,359)

| Modèle | F1-score |
|---|---|
| Régression logistique (baseline) | 0,511 |
| Random Forest (seuil optimisé) | 0,76 |
| **XGBoost (seuil optimisé, tuné)** | **0,767** |

Sur 100 visiteurs prédits convertis, ~80 le sont réellement, et le modèle retrouve 74 % des conversions réelles.

![Matrice de confusion finale](./final_confusion_matrix.png)

## 💡 Analyse des paramètres et recommandations

![Importance des variables](./feature_importance.png)

- **Nombre de pages visitées (~69 % de l'importance)** — le principal levier est comportemental, pas démographique. → *Recommandation : investir dans la découverte de contenu (recommandations d'articles, maillage interne) pour encourager l'engagement.*
- **Statut nouvel utilisateur (~9 %)** — les utilisateurs récurrents convertissent bien mieux. → *Recommandation : renforcer la rétention et le retargeting plutôt que la seule acquisition.*
- **Pays (~19 % cumulé)**, tiré par le contraste Chine vs reste du monde. → *Recommandation : creuser la cause racine (localisation, qualité du trafic) avant toute action marketing ciblée.*
- **Âge et source d'acquisition** : leviers secondaires (effet marginal, probablement déjà capté par les deux variables précédentes).

## 🛠️ Stack technique

- **Traitement & modélisation** : Pandas, NumPy, Scikit-learn, XGBoost
- **Visualisation** : Matplotlib, Seaborn
- **Sérialisation** : Joblib

## 📁 Structure du code

- `01_eda_preprocessing.py` — EDA, nettoyage, premiers modèles (baseline)
- `02_threshold_tuning.py` — optimisation du seuil de décision plutôt que rééquilibrage des classes
- `03_final_model_and_predictions.py` — tuning final, feature importance, prédictions sur le jeu de test

## 📬 Livrables

- Notebook / scripts d'EDA et de modélisation
- Fichier de prédictions soumis (`conversion_data_test_predictions.csv`)
- Mémoire d'analyse complet (contexte, méthodologie, résultats, recommandations)
