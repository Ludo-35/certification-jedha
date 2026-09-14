#!/usr/bin/env bash
# organize_repo.sh — Réorganise le repo FlyOnTime en local.
#
# Usage : place ce script à la RACINE de ton dossier de projet
# (là où se trouvent tous tes fichiers .py, Dockerfile*, requirements*.txt,
# les .parquet, etc.) puis lance :
#
#   chmod +x organize_repo.sh
#   ./organize_repo.sh
#
# Le script est idempotent : relancer plusieurs fois ne casse rien, il
# saute simplement ce qui a déjà été déplacé.

set -e

echo "🚀 Réorganisation du repo FlyOnTime..."

# ------------------------------------------------------------------
# 1. Création de l'arborescence cible
# ------------------------------------------------------------------
mkdir -p api dashboard training model cache legacy scripts data

# ------------------------------------------------------------------
# 2. Fonction utilitaire : déplace un fichier s'il existe encore à la racine
# ------------------------------------------------------------------
move_if_exists() {
    local src="$1"
    local dest="$2"
    if [ -f "$src" ]; then
        mv "$src" "$dest"
        echo "  ✓ $src -> $dest"
    fi
}

copy_if_exists() {
    local src="$1"
    local dest="$2"
    if [ -f "$src" ]; then
        cp "$src" "$dest"
        echo "  ✓ (copie) $src -> $dest"
    fi
}

# ------------------------------------------------------------------
# 3. api/ — service de prédiction
# ------------------------------------------------------------------
echo "📁 api/"
move_if_exists "api.py" "api/api.py"
move_if_exists "engine.py" "api/engine.py"
move_if_exists "feature_engineering.py" "api/feature_engineering.py"
move_if_exists "refresh_cache.py" "api/refresh_cache.py"
move_if_exists "requirements-api.txt" "api/requirements-api.txt"
move_if_exists "Dockerfile.api" "api/Dockerfile.api"

# ------------------------------------------------------------------
# 4. dashboard/ — app Streamlit consommant l'API
# ------------------------------------------------------------------
echo "📁 dashboard/"
move_if_exists "dashboard.py" "dashboard/dashboard.py"
move_if_exists "requirements-dashboard.txt" "dashboard/requirements-dashboard.txt"
move_if_exists "Dockerfile.dashboard" "dashboard/Dockerfile.dashboard"

# ------------------------------------------------------------------
# 5. training/ — entraînement du modèle (hors service déployé)
# ------------------------------------------------------------------
echo "📁 training/"
move_if_exists "train.py" "training/train.py"

# feature_engineering.py est utilisé À LA FOIS par l'API (déployée) et par
# l'entraînement (exécuté localement/ponctuellement). Comme api/ et
# training/ sont deux contextes d'exécution séparés, on en garde une COPIE
# dans training/ plutôt qu'un import cross-dossier fragile.
# ⚠️ Si tu modifies la feature engineering, réplique le changement dans les
# deux fichiers (ou migre plus tard vers un vrai package partagé).
copy_if_exists "api/feature_engineering.py" "training/feature_engineering.py"

if [ ! -f "training/requirements-training.txt" ]; then
    cat > training/requirements-training.txt << 'EOF'
pandas
pyarrow
xgboost
mlflow
boto3
psycopg2-binary
python-dotenv
EOF
    echo "  ✓ (créé) training/requirements-training.txt"
fi

# ------------------------------------------------------------------
# 6. legacy/ — anciens fichiers remplacés, conservés pour mémoire
# ------------------------------------------------------------------
echo "📁 legacy/"
move_if_exists "app.py" "legacy/app.py"
move_if_exists "appstlite.py" "legacy/appstlite.py"
move_if_exists "Dockerfile" "legacy/Dockerfile"
move_if_exists "requirements.txt" "legacy/requirements.txt"

# ------------------------------------------------------------------
# 7. scripts/ — utilitaires de dev, non déployés
# ------------------------------------------------------------------
echo "📁 scripts/"
move_if_exists "check_s3.py" "scripts/check_s3.py"
move_if_exists "debugs_cols.py" "scripts/debugs_cols.py"

# ------------------------------------------------------------------
# 8. data/ — jeux de données (à exclure de git si volumineux, voir .gitignore)
# ------------------------------------------------------------------
echo "📁 data/"
for f in df_airport_delays_179jours_20260329_224426.parquet \
         df_meteo_179jours_20260329_224426.parquet \
         df_mouvs_179jours_20260329_224426.parquet; do
    move_if_exists "$f" "data/$f"
done

# ------------------------------------------------------------------
# 9. model/ et cache/ — dossiers générés, vides pour l'instant
# ------------------------------------------------------------------
touch model/.gitkeep cache/.gitkeep

# ------------------------------------------------------------------
# 10. Fichiers restant à la racine : render.yaml, README.md, .gitignore,
#     .gitattributes — rien à faire, ils y sont déjà bien placés.
# ------------------------------------------------------------------
move_if_exists "_gitignore" ".gitignore"
move_if_exists "_gitattributes" ".gitattributes"

echo ""
echo "✅ Terminé. Arborescence obtenue :"
find . -maxdepth 2 -not -path './.git*' | sort
