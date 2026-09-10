import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# 1. Chargement des données
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "get_around_pricing_project.csv")

if not os.path.exists(data_path):
    data_path = os.path.join(current_dir, "..", "get_around_pricing_project.csv")

df = pd.read_csv(data_path)

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 2. Définition des variables explicatives (X) et de la cible (y)
target = 'rental_price_per_day'
X = df.drop(columns=[target])
y = df[target]

categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 3. Pipeline de prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

# 4. Pipeline du modèle (Prétraitement + Modèle)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Entraînement
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Entraînement du modèle en cours...")
model_pipeline.fit(X_train, y_train)

# 6. Évaluation
y_pred = model_pipeline.predict(X_test)
print("\n--- PERFORMANCES DU MODÈLE ---")
print(f"R² Score : {r2_score(y_test, y_pred):.4f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.2f} €")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f} €")

# 7. Sauvegarde du modèle
model_file = os.path.join(current_dir, "model.joblib")
joblib.dump(model_pipeline, model_file)
print(f"\nModèle enregistré avec succès : {model_file}")