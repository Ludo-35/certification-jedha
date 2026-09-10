# 1. Utiliser une image Python officielle (version 'slim' pour la légèreté)
FROM python:3.10-slim

# 2. Installer les dépendances système (nécessaires pour Prophet et Postgres)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 4. Copier le fichier des dépendances en premier (optimise le cache Docker)
COPY requirements.txt .

# 5. Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copier tout le reste de ton code (app.py, engine.py, etc.)
COPY . .

# 7. Exposer le port par défaut de Streamlit (8501)
EXPOSE 8501

# Force le port 7860 pour Hugging Face
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
