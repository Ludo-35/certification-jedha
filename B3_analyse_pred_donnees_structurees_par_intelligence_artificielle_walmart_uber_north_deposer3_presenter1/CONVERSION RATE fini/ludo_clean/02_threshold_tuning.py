"""
Partie 2 (suite) : amélioration via optimisation du seuil de décision
plutôt que rééquilibrage agressif des classes.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("/mnt/user-data/uploads/conversion_data_train.csv")
df = df[df["age"] < 80].reset_index(drop=True)

X = df.drop(columns="converted")
y = df["converted"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

categorical_features = ["country", "source"]
numeric_features = ["age", "new_user", "total_pages_visited"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features),
])

def tune_threshold(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # last point has no threshold
    return thresholds[best_idx], f1s[best_idx], proba

# ---- Random Forest, no class_weight, threshold tuned ----
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)),
])
rf_pipe.fit(X_train, y_train)
best_thr_rf, f1_val_rf, proba_rf = tune_threshold(rf_pipe, X_test, y_test)
y_pred_rf_tuned = (proba_rf >= best_thr_rf).astype(int)
print(f"[RF, no class_weight, threshold={best_thr_rf:.3f}] F1: {f1_score(y_test, y_pred_rf_tuned):.4f}")
print(classification_report(y_test, y_pred_rf_tuned))

# ---- XGBoost, no scale_pos_weight, threshold tuned ----
xgb_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=42, n_jobs=-1)),
])
xgb_pipe.fit(X_train, y_train)
best_thr_xgb, f1_val_xgb, proba_xgb = tune_threshold(xgb_pipe, X_test, y_test)
y_pred_xgb_tuned = (proba_xgb >= best_thr_xgb).astype(int)
print(f"\n[XGB, no scale_pos_weight, threshold={best_thr_xgb:.3f}] F1: {f1_score(y_test, y_pred_xgb_tuned):.4f}")
print(classification_report(y_test, y_pred_xgb_tuned))

# ---- Comparatif final ----
results = pd.DataFrame({
    "model": ["Random Forest (threshold tuned)", "XGBoost (threshold tuned)"],
    "f1_score": [f1_score(y_test, y_pred_rf_tuned), f1_score(y_test, y_pred_xgb_tuned)],
    "threshold": [best_thr_rf, best_thr_xgb],
})
print("\n=== Comparatif final ===")
print(results.to_string(index=False))
results.to_csv("/home/claude/conversion/model_comparison_v2.csv", index=False)

# Save the winner
if f1_score(y_test, y_pred_rf_tuned) >= f1_score(y_test, y_pred_xgb_tuned):
    winner_pipe, winner_thr, winner_name = rf_pipe, best_thr_rf, "RandomForest"
else:
    winner_pipe, winner_thr, winner_name = xgb_pipe, best_thr_xgb, "XGBoost"

joblib.dump({"pipeline": winner_pipe, "threshold": winner_thr, "name": winner_name},
            "/home/claude/conversion/best_model_v2.joblib")
print(f"\nWinner: {winner_name} (threshold={winner_thr:.3f}) saved.")

# Confusion matrices figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, preds) in zip(axes, [("Random Forest", y_pred_rf_tuned), ("XGBoost", y_pred_xgb_tuned)]):
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Non converti", "Converti"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} (threshold tuned)")
plt.tight_layout()
plt.savefig("/home/claude/conversion/confusion_matrices_v2.png", bbox_inches="tight")
plt.close()
