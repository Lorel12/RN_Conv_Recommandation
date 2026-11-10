# Système de recommandation de films avec LightGCN

Prototype de recommandation personnalisée de films basé sur le filtrage
collaboratif et le jeu de données **MovieLens Latest Small**. Le projet
comprend une analyse exploratoire, l'entraînement et l'évaluation d'un modèle
LightGCN, ainsi qu'une application web Flask pour consulter des recommandations.

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Technologies](#technologies)
- [Structure du projet](#structure-du-projet)
- [Données](#données)
- [Installation](#installation)
- [Entraîner le modèle](#entraîner-le-modèle)
- [Lancer l'application](#lancer-lapplication)
- [Évaluation](#évaluation)
- [Limites](#limites)
- [Améliorations possibles](#améliorations-possibles)
- [Références](#références)

## Fonctionnalités

- Analyse exploratoire des notes et des interactions utilisateur-film.
- Filtrage des notes pour retenir les interactions positives (note supérieure
  ou égale à 4).
- Construction d'un graphe biparti utilisateurs-films et normalisation de sa
  matrice d'adjacence.
- Apprentissage de représentations (embeddings) avec LightGCN et PyTorch.
- Entraînement pairwise avec la perte Bayesian Personalized Ranking (BPR),
  échantillonnage négatif et régularisation L2.
- Génération de recommandations personnalisées en classant les films par
  produit scalaire entre embeddings, tout en écartant les films déjà vus.
- Évaluation hors ligne avec Recall@K et NDCG@K.
- Interface Flask avec sélection d'un utilisateur, affichage de son historique
  et génération de recommandations, ainsi qu'un tableau de bord de métriques.

## Technologies

- Python
- PyTorch
- NumPy, Pandas et SciPy
- Matplotlib et Seaborn
- Flask et templates HTML
- Jupyter Notebook

Les versions minimales et la version de Flask sont indiquées dans
[`requirements.txt`](requirements.txt).

## Structure du projet

```text
.
├── app.py                       # Application web Flask
├── data_utils.py                # Chargement et préparation des données, graphe
├── evaluation.py                # Calcul de Recall@K et NDCG@K
├── model.py                     # Architecture LightGCN
├── recommandations.py           # Génération des recommandations
├── train.py                     # Fonctions de perte et d'entraînement
├── light_gcn.ipynb              # Analyse, entraînement, évaluation et artefacts
├── lightgcn_artifacts.pkl       # Embeddings et données nécessaires à l'application
├── requirements.txt             # Dépendances Python
├── ml-latest-small/             # Données MovieLens utilisées par le projet
├── templates/
│   ├── index.html               # Interface de recommandation
│   └── admin.html               # Tableau de bord
└── static/Images/               # Graphiques affichés dans le tableau de bord
```

Le notebook est le point d'entrée de l'analyse et de l'entraînement. Le fichier
`train.py` contient des fonctions réutilisées par le notebook ; ce n'est pas à
lui seul un script d'entraînement exécutable.

## Données

Le projet utilise le jeu de données **MovieLens Latest Small**, inclus dans
`ml-latest-small/`. Les fichiers principaux sont `ratings.csv` et `movies.csv`.
Le notebook analyse les notes et les interactions, puis le pipeline conserve
les notes supérieures ou égales à 4 pour construire les interactions positives.

Dans l'exécution documentée dans le notebook, le filtrage a retenu **48 580
interactions**, concernant **609 utilisateurs** et **6 298 films**. Ces nombres
décrivent ce jeu de données et ce seuil précis ; ils ne constituent pas une
garantie de performance du modèle.

Le jeu de données contient également son fichier d'information
[`ml-latest-small/README.txt`](ml-latest-small/README.txt). Consultez-le ainsi
que les conditions d'utilisation de MovieLens avant toute redistribution ou
utilisation autre que pédagogique.

## Installation

Depuis la racine du dépôt, créez un environnement virtuel et installez les
dépendances :

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloque l'activation du script, vous pouvez appeler directement
l'interpréteur de l'environnement virtuel sans l'activer :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Le projet a été développé pour une exécution locale. Une installation PyTorch
compatible avec votre système est nécessaire ; consultez la page d'installation
officielle de PyTorch si la commande ci-dessus ne sélectionne pas la bonne
version pour votre matériel.

## Entraîner le modèle

L'entraînement, l'évaluation et la génération des artefacts sont réalisés dans
[`light_gcn.ipynb`](light_gcn.ipynb) :

1. Installez Jupyter Notebook s'il n'est pas déjà disponible dans
   l'environnement :

   ```powershell
   python -m pip install notebook
   ```

2. Démarrez Jupyter depuis la racine du dépôt :

   ```powershell
   jupyter notebook
   ```

3. Ouvrez `light_gcn.ipynb` et exécutez les cellules dans l'ordre.
4. Le notebook charge les données de `ml-latest-small/`, prépare le graphe,
   entraîne le modèle et calcule les métriques.
5. Les paramètres présents dans le notebook comprennent une dimension
   d'embedding de 64, 3 couches LightGCN, 20 époques et une évaluation tous les
   2 époques. Ils peuvent être ajustés dans les cellules de configuration.
6. Le notebook enregistre les artefacts appris dans
   `lightgcn_artifacts.pkl` et produit des graphiques dans `Images/`.

L'application Flask lit `lightgcn_artifacts.pkl` au démarrage. Il faut donc que
ce fichier soit présent et compatible avec la version du code ; réexécuter le
notebook permet de le régénérer.

## Lancer l'application

Après l'installation des dépendances et la présence de
`lightgcn_artifacts.pkl`, lancez depuis la racine du dépôt :

```powershell
python app.py
```

Ouvrez ensuite [http://127.0.0.1:5000](http://127.0.0.1:5000).

- **Plateforme** (`/`) : choisissez un utilisateur et lancez la génération de
  recommandations. L'application affiche jusqu'à 20 films et un aperçu de
  l'historique positif de l'utilisateur.
- **Dashboard technique** (`/admin`) : consultez les métriques disponibles,
  les statistiques et les graphiques du modèle.

Le serveur Flask est configuré en mode debug dans le code : cette configuration
est réservée au développement local, pas à un déploiement public.

## Évaluation

Le notebook et `evaluation.py` évaluent les recommandations sur un découpage
entraînement/test par utilisateur. Les interactions de test servent à mesurer :

- **Recall@K** : proportion des films pertinents retrouvés parmi les K premiers
  résultats.
- **NDCG@K** : qualité du classement, qui valorise davantage les films
  pertinents placés en haut de la liste.

Les films présents dans l'ensemble d'entraînement de chaque utilisateur sont
masqués avant le classement. Les métriques affichées dépendent du jeu de
données, du découpage et des paramètres utilisés ; relancez le notebook pour
produire et vérifier les résultats de votre exécution.

## Limites

- Le modèle dépend des interactions historiques ; il ne traite pas directement
  le problème de démarrage à froid pour un nouvel utilisateur ou un nouveau film.
- Le seuil de note (≥ 4) transforme les notes en interactions positives et ne
  prend pas en compte toute la nuance des préférences.
- Les performances sont mesurées hors ligne sur MovieLens et ne démontrent pas
  à elles seules la qualité des recommandations en production.
- L'application est un prototype local ; elle ne comprend pas de dispositif
  d'authentification ni de configuration de déploiement public.

## Améliorations possibles

- Ajouter des tests automatisés pour les fonctions de préparation, de
  recommandation et d'évaluation.
- Rendre les paramètres d'entraînement configurables depuis une commande ou un
  fichier de configuration.
- Ajouter un parcours de déploiement avec paramètres de production et
  vérification des artefacts.
- Étudier des stratégies de démarrage à froid et comparer LightGCN à des
  approches de référence.
- Documenter les scores obtenus et les conditions exactes de chaque expérience.

## Références

- He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for
  Recommendation*, SIGIR 2020.
- [Page du jeu de données MovieLens](https://grouplens.org/datasets/movielens/)
- [Documentation PyTorch](https://pytorch.org/docs/)
- [Documentation Flask](https://flask.palletsprojects.com/)
