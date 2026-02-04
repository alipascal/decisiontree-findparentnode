# Identification de la racine d'un arbre de décision

Programme Python qui, à partir de n'importe quel fichier contenant un jeu de données pour l'entraînement des arbres de décision, retourne la racine de l'arbre de décision construit selon le principe de minimisation de l'entropie.

## Fonctionnalités

* Calcul d'entropie
* Validation *témoin* avec `scipy` et `sklearn`
* Support des chaînes de caractères et entiers (numérisation automatique)
* Visualisation optionnelle avec `matplotlib`

## Résultats / Sortie

exemple :
```
Racine choisie par le programme : ('P3', 0.3605)
Racine scipy (témoin) : ('P3', 0.2499)
Arbre sklearn (témoin) :
|--- P3 ...
```

Le programme retourne la racine et deux résultats témoins pour validation.

## Utilisation

```shell
python program.py                          # Fichier par défaut (example.csv)
python program.py fichier.csv yes          # Avec affichage graphique
python program.py --file exo1.csv --plot yes
```

**Format CSV requis :**
```
a b c class
1 3 2 A
0 0 2 B
...
```

## Installation

```shell
pip install -r requirements.txt
```

Dépendances :
  - `pandas`,
  - `numpy`,
  - `scikit-learn`
  - `matplotlib`
  - `scipy`
