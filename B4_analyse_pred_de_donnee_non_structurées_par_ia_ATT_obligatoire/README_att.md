# 📵 AT&T — Détecteur de spam SMS

**Bloc de certification :** B4 — Analyse prédictive de données non structurées par IA

## 🎯 Contexte

Pour **AT&T**, construire un détecteur de spam SMS capable de classifier un message comme spam ou non, à partir du **seul contenu textuel** du SMS (dataset SMS Spam Collection).

## 🎯 Objectif

Livrer un notebook présentant le prétraitement des données textuelles, l'entraînement de modèles de **deep learning**, et une comparaison claire des résultats obtenus — en explorant notamment le **transfer learning**, particulièrement adapté au faible volume de données disponible.

## 🛠️ Approche — 5 modèles comparés

1. **Baseline** : Embedding + couches Dense
2. **RNN maison (v1)** : architecture récurrente construite from scratch
3. **RNN maison (v2)** : variante de l'architecture récurrente
4. **DistilBERT générique** en transfer learning
5. **DistilBERT spécialisé spam** (+ variante avec feature engineering additionnel)

Cette progression permet de comparer une approche simple (embedding + dense), des architectures récurrentes classiques, puis des modèles pré-entraînés de type transformer adaptés par transfer learning — cohérent avec la piste suggérée dans l'énoncé (démarrer simple, envisager le transfer learning vu le faible volume de données).

## 🛠️ Stack technique

- **Deep Learning** : TensorFlow/Keras ou PyTorch
- **NLP** : Embeddings, RNN, DistilBERT (Hugging Face Transformers)
- **Traitement** : Pandas, NumPy

## 📬 Livrables

- Notebook complet : prétraitement, entraînement des 5 modèles, résultats comparés
- Mémoire associé au projet
