# ✈️ Kayak — Recommandation de destinations de voyage

**Bloc de certification :** B1 — Construction et alimentation d'une infrastructure de gestion de données

## 🎯 Contexte

Kayak (groupe Booking Holdings) souhaite lancer une application recommandant des destinations de vacances en France, basée sur des données réelles de météo et d'hôtellerie, afin de rassurer des utilisateurs sceptiques quant aux contenus non sourcés.

## 🎯 Objectif du projet

Construire, à partir de zéro, l'infrastructure de données nécessaire à cette application :
1. Récupérer les coordonnées GPS des villes cibles.
2. Collecter les données météo de ces villes.
3. Scraper les données hôtelières correspondantes.
4. Stocker l'ensemble dans un **data lake** (S3).
5. Nettoyer et charger les données dans un **data warehouse** (base SQL).

**Périmètre :** 35 villes touristiques incontournables de France (sélection basée sur un classement One Week In.com).

## 🏗️ Pipeline réalisé

### 1. Coordonnées GPS
Récupération des coordonnées des 35 villes cibles via l'**API Nominatim**.

### 2. Données météo
Appel à l'**API OpenWeatherMap** (forecast 7 jours) pour chaque ville : température moyenne, probabilité et volume de précipitations. Calcul d'un score de "beau temps" pour identifier le **top 5 des meilleures destinations** à date.

### 3. Visualisation géographique
Cartes interactives **Plotly** (`scatter_map`) affichant le top 5 des destinations les plus ensoleillées, exportées en HTML.

### 4. Scraping hôtelier
Scraping de **Booking.com** (Selenium + webdriver-manager) pour récupérer, par ville : nom de l'hôtel, URL, coordonnées GPS, note utilisateurs, description textuelle.

### 5. Data Lake — AWS S3
Fusion des données villes (météo/GPS) et hôtels dans un dataframe unique, stocké en CSV sur un bucket **S3**.

### 6. ETL vers Data Warehouse
Extraction des données depuis S3, transformation, puis chargement dans une base **SQL (AWS RDS)** via SQLAlchemy — pour une consommation propre par l'équipe d'analyse en aval.

## 🛠️ Stack technique

- **Collecte** : Requests (API Nominatim, OpenWeatherMap), Selenium (scraping Booking.com)
- **Traitement** : Pandas, NumPy
- **Visualisation** : Plotly
- **Cloud/Stockage** : AWS S3 (data lake), AWS RDS (data warehouse SQL)
- **Sécurité** : variables d'environnement via `python-dotenv` pour les identifiants AWS/RDS

## 📬 Livrables

- Fichier CSV enrichi (météo + hôtels) stocké sur S3
- Base de données SQL interrogeable sur RDS
- Carte des 5 meilleures destinations
- Carte des hôtels les mieux notés de la région sélectionnée
