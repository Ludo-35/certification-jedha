"""
Défi taux de conversion - Data Science Weekly
Partie 1 & 2 : EDA, preprocessing, baseline et amélioration du modèle
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import joblib

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# -----------------------------------------------------------------------
# 1. Chargement des données
# -----------------------------------------------------------------------
df = pd.read_csv("/mnt/user-data/uploads/conversion_data_train.csv")
print("Shape:", df.shape)
print(df.head())
print(df["converted"].value_counts(normalize=True))

# -----------------------------------------------------------------------
# 2. EDA - figures
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Distribution de la cible
df["converted"].value_counts().plot(kind="bar", ax=axes[0, 0], color=["#4C72B0", "#DD8452"])
axes[0, 0].set_title("Distribution de la cible (converted)")
axes[0, 0].set_xticklabels(["Non converti", "Converti"], rotation=0)

# Taux de conversion par pays
df.groupby("country")["converted"].mean().sort_values().plot(kind="barh", ax=axes[0, 1], color="#4C72B0")
axes[0, 1].set_title("Taux de conversion par pays")

# Taux de conversion par source
df.groupby("source")["converted"].mean().sort_values().plot(kind="barh", ax=axes[0, 2], color="#55A868")
axes[0, 2].set_title("Taux de conversion par source")

# Taux de conversion par new_user
df.groupby("new_user")["converted"].mean().plot(kind="bar", ax=axes[1, 0], color="#C44E52")
axes[1, 0].set_title("Taux de conversion : nouvel utilisateur vs existant")
axes[1, 0].set_xticklabels(["Utilisateur existant", "Nouvel utilisateur"], rotation=0)

# Pages visitées vs conversion
sns.boxplot(data=df, x="converted", y="total_pages_visited", ax=axes[1, 1])
axes[1, 1].set_title("Pages visitées vs conversion")
axes[1, 1].set_xticklabels(["Non converti", "Converti"])

# Age distribution
sns.histplot(df[df["age"] < 80]["age"], bins=40, ax=axes[1, 2], color="#8172B2")
axes[1, 2].set_title("Distribution de l'âge (<80 ans)")

plt.tight_layout()
plt.savefig("/home/claude/conversion/eda_overview.png", bbox_inches="tight")
plt.close()
print("EDA figure saved.")

# -----------------------------------------------------------------------
# 3. Nettoyage
# -----------------------------------------------------------------------
# Suppression des 2 lignes avec âge aberrant (111 et 123 ans)
before = len(df)
df = df[df["age"] < 80].reset_index(drop=True)
print(f"Removed {before - len(df)} age outlier rows")

# -----------------------------------------------------------------------
# 4. Split train/test stratifié
# -----------------------------------------------------------------------
X = df.drop(columns="converted")
y = df["converted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -----------------------------------------------------------------------
# 5. Preprocessing pipeline
# -----------------------------------------------------------------------
categorical_features = ["country", "source"]
numeric_features = ["age", "new_user", "total_pages_visited"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ]
)

# -----------------------------------------------------------------------
# 6. Baseline : régression logistique
# -----------------------------------------------------------------------
logreg_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
])
logreg_pipe.fit(X_train, y_train)
y_pred_lr = logreg_pipe.predict(X_test)
f1_lr = f1_score(y_test, y_pred_lr)
print(f"\n[Baseline] Logistic Regression F1: {f1_lr:.4f}")
print(classification_report(y_test, y_pred_lr))

# -----------------------------------------------------------------------
# 7. Random Forest (class_weight balanced)
# -----------------------------------------------------------------------
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )),
])
rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)
f1_rf = f1_score(y_test, y_pred_rf)
print(f"\n[Random Forest] F1: {f1_rf:.4f}")
print(classification_report(y_test, y_pred_rf))

# -----------------------------------------------------------------------
# 8. XGBoost (scale_pos_weight)
# -----------------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42, n_jobs=-1
    )),
])
xgb_pipe.fit(X_train, y_train)
y_pred_xgb = xgb_pipe.predict(X_test)
f1_xgb = f1_score(y_test, y_pred_xgb)
print(f"\n[XGBoost] F1: {f1_xgb:.4f}")
print(classification_report(y_test, y_pred_xgb))

# -----------------------------------------------------------------------
# 9. Grid search rapide sur le meilleur candidat (XGBoost)
# -----------------------------------------------------------------------
param_grid = {
    "clf__max_depth": [3, 4, 5],
    "clf__n_estimators": [200, 300, 400],
    "clf__learning_rate": [0.05, 0.1, 0.2],
}
grid = GridSearchCV(xgb_pipe, param_grid, scoring="f1", cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print("\nBest params:", grid.best_params_)
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)
f1_best = f1_score(y_test, y_pred_best)
print(f"[XGBoost tuned] F1: {f1_best:.4f}")
print(classification_report(y_test, y_pred_best))

# -----------------------------------------------------------------------
# 10. Comparatif des modèles + matrices de confusion
# -----------------------------------------------------------------------
results = pd.DataFrame({
    "model": ["Logistic Regression", "Random Forest", "XGBoost", "XGBoost (tuned)"],
    "f1_score": [f1_lr, f1_rf, f1_xgb, f1_best],
})
results = results.sort_values("f1_score", ascending=False)
print("\n=== Comparatif des modèles ===")
print(results.to_string(index=False))
results.to_csv("/home/claude/conversion/model_comparison.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (name, preds) in zip(
    axes,
    [("Logistic Regression", y_pred_lr), ("Random Forest", y_pred_rf),
     ("XGBoost", y_pred_xgb), ("XGBoost (tuned)", y_pred_best)]
):
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Non converti", "Converti"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig("/home/claude/conversion/confusion_matrices.png", bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------
# 11. Sauvegarde du meilleur modèle
# -----------------------------------------------------------------------
joblib.dump(best_model, "/home/claude/conversion/best_model.joblib")
print("\nBest model saved.")
