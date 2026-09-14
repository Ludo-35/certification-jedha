# Detecteur de spam SMS - Projet AT&T

## Presentation

AT&T souhaite detecter automatiquement les SMS indesirables (spam) recus par ses utilisateurs, en se basant uniquement sur le contenu du message. Il s'agit d'un probleme de classification binaire de texte : a partir d'un SMS, predire s'il s'agit d'un spam ou d'un message legitime (ham).

Le jeu de donnees utilise est le SMS Spam Collection : 5572 SMS en anglais, etiquetes ham ou spam. Le jeu de donnees est desequilibre (environ 13 pourcent de spams).

Deux modeles sont construits et compares dans le notebook :

1. Un modele de reference (baseline), simple : embedding, pooling, couches denses. Entraine de zero.
2. Un modele de transfer learning, base sur DistilBERT deja pre-entraine pour la detection de spam SMS (modele recupere sur Hugging Face). Ses poids sont geles, seule une petite tete de classification est entrainee.

Les deux modeles sont evalues sur un ensemble de validation isole avant tout retraitement des donnees, afin de garder une mesure honnete de leurs performances.

## Resultats

| Modele | Accuracy | Precision | Recall |
|---|---|---|---|
| 1. Baseline (embedding + dense) | 0.9868 | 0.9633 | 0.9375 |
| 2. Transfer learning (DistilBERT) | 0.9964 | 1.0000 | 0.9732 |

Le modele retenu est le modele 2 (DistilBERT), pour ses meilleures performances sur les trois metriques.

## Structure du projet

- `AT_T_minimal.ipynb` : notebook principal (pretraitement, entrainement, evaluation des deux modeles)
- `spam.csv` : jeu de donnees

## Installation

Le projet a ete teste avec Python 3.11, dans un environnement conda dedie.

Creation de l'environnement :

```
conda create -n att_spam python=3.11
conda activate att_spam
```

Installation des librairies necessaires :

```
pip install "tensorflow==2.15.0" "numpy<2.0" tf-keras "transformers==4.30.2" scikit-learn matplotlib spacy pandas
python -m spacy download en_core_web_md
```

Librairies utilisees dans le notebook :

- `numpy`, `pandas` : manipulation de donnees
- `matplotlib` : visualisation (matrices de confusion)
- `spacy` (+ modele `en_core_web_md`) : lemmatisation du texte
- `tensorflow` et `tf-keras` : construction et entrainement des modeles de deep learning
- `transformers` : chargement du modele pre-entraine DistilBERT et de son tokenizer
- `scikit-learn` : separation des donnees et calcul des metriques

Remarque : les versions de `tensorflow` et `transformers` sont volontairement figees, certaines combinaisons de versions plus recentes provoquant des erreurs d'incompatibilite.

Un GPU n'est pas obligatoire pour faire tourner ce notebook. Il accelere l'entrainement du modele DistilBERT (modele 2), mais le notebook reste utilisable sur CPU, avec un temps d'entrainement plus long sur cette partie.

## Execution

Une fois l'environnement installe et active, ouvrir `AT_T_minimal.ipynb` dans Jupyter Lab ou VS Code, selectionner l'interpreteur Python correspondant a l'environnement `att_spam`, puis executer les cellules dans l'ordre.

## Credits

Modele pre-entraine DistilBERT : Victor Sanh, Lysandre Debut, Julien Chaumond et Thomas Wolf (papier : https://arxiv.org/abs/1910.01108)

Modele DistilBERT specialise spam SMS : Manirathinam21, disponible sur Hugging Face (https://huggingface.co/Manirathinam21/DistilBert_SMSSpam_classifier)
