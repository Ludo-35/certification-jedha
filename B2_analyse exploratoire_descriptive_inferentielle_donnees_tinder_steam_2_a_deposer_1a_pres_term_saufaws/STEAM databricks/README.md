
Link to the published databricks [project](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/80626097791223/696301889738918/51873322286668/latest.html)

# Steam Videogames Market Analysis — Ubisoft Case Study

Databricks • PySpark • Big Data Analytics

---

## Résumé exécutif

Ce projet analyse le marché des jeux vidéo sur Steam à partir d’un dataset massif JSON semi-structuré afin d’identifier les facteurs clés de succès et de formuler des recommandations stratégiques pour un éditeur premium comme Ubisoft.

Il démontre une maîtrise complète du pipeline data, depuis l’ingestion de données complexes jusqu’à la production d’insights business actionnables, en utilisant Databricks et PySpark.

---

## Objectifs du projet

* Analyser plus de 70 000 jeux Steam à grande échelle
* Construire un pipeline PySpark robuste et reproductible
* Extraire des indicateurs fiables de popularité et de qualité
* Étudier l’impact du prix, des genres, des plateformes et du contexte COVID
* Proposer une recommandation produit réaliste et argumentée pour Ubisoft

---

## Stack technique

* Apache Spark (PySpark)
* Databricks
* Stockage S3 (dataset public Jedha)
* Visualisations natives Databricks

---

## Données

Source :
`s3://full-stack-bigdata-datasets/Big_Data/Project_Steam/steam_game_output.json`

Format :
JSON semi-structuré avec structures imbriquées (structs), listes (arrays) et champs hétérogènes (prix, dates, plateformes, tags).

---

## Architecture du pipeline

### 1. Ingestion et inspection

* Lecture JSON distribuée avec Spark
* Inférence et validation du schéma
* Analyse exploratoire initiale

### 2. Flattening et normalisation

* Extraction explicite des champs imbriqués
* Renommage cohérent des colonnes
* Suppression de toute ambiguïté d’affichage Databricks

### 3. Nettoyage et feature engineering

**Prix**

* Données brutes en centimes
* Conversion explicite en euros pour l’analyse business
* Conservation des valeurs originales pour éviter les erreurs numériques

**Dates**

* Nettoyage par expressions régulières
* Normalisation des formats hétérogènes
* Parsing multi-format sécurisé
* Gestion robuste des valeurs invalides sans exception Spark

**Reviews**

* Calcul de `total_reviews`
* Calcul de `positive_ratio` comme indicateur normalisé de satisfaction

**Temps**

* Extraction de l’année et du mois
* Création d’une variable `covid_period` (pre / covid / post)

**Langues**

* Découpage et comptage du nombre de langues supportées

### 4. Explosion des genres

* Transformation 1 jeu → N genres
* Analyse granulaire par genre
* Préparation d’une table dédiée pour les analyses croisées

---

## Analyses réalisées

### Analyse macro

* Répartition des jeux par plateforme
* Évolution annuelle des sorties
* Impact du COVID sur la production
* Distribution des prix
* Éditeurs les plus actifs
* Accessibilité linguistique

### Analyse par genre

* Genres les plus représentés
* Popularité mesurée par volume de reviews
* Qualité mesurée par ratio de reviews positives
* Identification des genres à fort potentiel
* Analyse des blockbusters Steam

### Analyse plateformes × genres

* Répartition Windows / macOS / Linux
* Identification des genres performants sur Linux et Steam Deck
* Corrélation multi-plateforme et satisfaction utilisateur

---

## Principaux résultats

* Steam est dominé en volume par les jeux indépendants
* Le prix médian du marché est bas (environ 6 €)
* Les blockbusters combinent profondeur mécanique, rejouabilité et multi-genre
* Les genres Action, RPG, Simulation et Strategy concentrent la majorité des succès durables
* Les jeux multi-plateformes obtiennent de meilleures évaluations

---

## Recommandation stratégique pour Ubisoft

Développer un jeu Action-RPG ou Simulation systémique avec les caractéristiques suivantes :

* Forte rejouabilité
* Mécaniques ouvertes et évolutives
* Support du modding (Steam Workshop)
* Cible PC prioritaire (Windows), compatibilité macOS et Steam Deck
* Positionnement premium entre 39 € et 59 €

Cette stratégie est alignée avec les jeux les plus performants sur Steam en termes de longévité, de satisfaction et d’engagement communautaire.

---

## Bonnes pratiques et qualité du code

* Utilisation exclusive de transformations PySpark natives
* Aucune UDF
* Pas de collect ou show non maîtrisé
* Parsing sécurisé sans erreurs Spark
* Code structuré, commenté et reproductible
* Notebook prêt pour publication Databricks

---

## Compétences démontrées

**Data Engineering**

* Traitement de données semi-structurées à grande échelle
* Optimisation Spark
* Pipeline robuste et scalable
* Gestion des schémas complexes

**Data Science / Analytics**

* Feature engineering métier
* Analyse exploratoire approfondie
* Visualisation orientée décision
* Storytelling data pour un public non technique

**Business**

* Traduction des données en décisions produit
* Recommandation stratégique réaliste
* Compréhension des dynamiques marché

---

## Auteur

Faycel Faddaoui
Projet de fin de formation Jedha
Spécialisation Data Engineering / Data Science / Analytics