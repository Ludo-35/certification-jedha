"""
Partie 2 (fin), 3 & 4 : tuning fin, feature importance, prédictions test, recommandations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report, make_scorer
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

def best_f1_threshold(y_true, proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    idx = np.argmax(f1s[:-1])
    return thresholds[idx], f1s[idx]

# -----------------------------------------------------------------------
# Grid search XGBoost (no scale_pos_weight; threshold optimized after)
# -----------------------------------------------------------------------
xgb_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)),
])
param_grid = {
    "clf__max_depth": [4, 6],
    "clf__n_estimators": [200, 300],
    "clf__learning_rate": [0.05, 0.1],
}
grid = GridSearchCV(xgb_pipe, param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
best_model = grid.best_estimator_

proba_test = best_model.predict_proba(X_test)[:, 1]
thr, f1_val = best_f1_threshold(y_test, proba_test)
y_pred = (proba_test >= thr).astype(int)
print(f"\n[Final XGBoost] threshold={thr:.3f}  F1={f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Non converti", "Converti"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, colorbar=False)
ax.set_title(f"Modèle final (F1={f1_score(y_test, y_pred):.3f})")
plt.tight_layout()
plt.savefig("/home/claude/conversion/final_confusion_matrix.png", bbox_inches="tight")
plt.close()

# -----------------------------------------------------------------------
# Feature importance (Partie 4)
# -----------------------------------------------------------------------
ohe = best_model.named_steps["prep"].named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(categorical_features)
feature_names = list(cat_names) + numeric_features
importances = best_model.named_steps["clf"].feature_importances_

imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(imp_df["feature"], imp_df["importance"], color="#4C72B0")
ax.set_title("Importance des variables (XGBoost)")
plt.tight_layout()
plt.savefig("/home/claude/conversion/feature_importance.png", bbox_inches="tight")
plt.close()
print("\n=== Feature importance ===")
print(imp_df.sort_values("importance", ascending=False).to_string(index=False))

# -----------------------------------------------------------------------
# Refit on 100% of cleaned training data before predicting on real test set
# -----------------------------------------------------------------------
final_pipe = grid.best_estimator_
final_pipe.fit(X, y)  # refit on all available labeled data

# -----------------------------------------------------------------------
# Predictions on conversion_data_test.csv
# -----------------------------------------------------------------------
test_df = pd.read_csv("/mnt/user-data/uploads/conversion_data_test.csv")
print("\nTest file shape:", test_df.shape)
print(test_df.head())

proba_final = final_pipe.predict_proba(test_df)[:, 1]
preds_final = (proba_final >= thr).astype(int)

submission = pd.DataFrame({"converted": preds_final})
submission.to_csv("/home/claude/conversion/conversion_data_test_predictions.csv", index=False)
print(f"\nPredicted conversion rate on test set: {preds_final.mean():.4f}")
print("Predictions saved.")

joblib.dump({"pipeline": final_pipe, "threshold": thr}, "/home/claude/conversion/final_model.joblib")
