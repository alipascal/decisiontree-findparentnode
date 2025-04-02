"""
@author Alicia TCHEMO
@date 2025-03-20
Apprentissage Machine - M1 INFO DCI - Université Paris-Cité

Identification du noeud parent d'un arbre de décision selon du principe de minimisation de l'entropie

"""

import pandas as pd
import numpy as np
from math import log2

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt

from scipy.stats import entropy


def entropy_calcul(probabilites: list) -> float:
    """
    Calcule l'entropie selon des probabilités données, tel que l'entropie
    E(X) = − ∑ pi x log2(pi)

    :param probabilites: list(float) Liste de propabilités
    :return: L'entropie calculé
    """
    return -sum(p * log2(p) for p in probabilites if p > 0)



def findparentnode(file: str, raw: bool = True) -> str:
    """
    Trouve le noeud parent d'un fichier donné.

    :param file: Nom du fichier
    :param raw: Indique si le chemin doit être retourné sous forme brute
    :return: Nœud parent de l'arbre de décision déduit grace au fichier
    """

    df = pd.read_csv(file, sep=' ')

    classification = df.columns[-1]

    eaps = []

    if df[classification].dtype != 'int64':
        # Numérisation de la colonne classification
        d = {elem: index for index, elem in enumerate(pd.unique(df[classification]))}
        df[classification] = df[classification].map(d)

    # Boucle for sur toutes les colonnes (sauf celle de la classification)
    for i in range(df.shape[1] - 1):
        eap = []
        colonne = df.columns[i]

        # Numérisation de la colonne si nécessaire
        if df[colonne].dtype != 'int64':
            d = {elem: index for index, elem in enumerate(pd.unique(df[colonne]))}
            df[colonne] = df[colonne].map(d)

        df_colonne = df[[colonne, classification]]
        denominateur = df[colonne].value_counts().sum() # N+P
        df_count_classification = df_colonne.groupby(colonne).value_counts() # Ré-organisation de la table 
        
        for j in range(len(df[colonne].value_counts())):
            numerateur = df_count_classification.loc[j].sum()
            probabilites = list( float(elem / numerateur) for elem in df_count_classification.loc[j].tolist() )
            entropie = None
            if raw:
                entropie = entropy_calcul(probabilites)
            else:
                entropie = entropy(probabilites)
            eap.append(entropie * numerateur / denominateur)
        
        eaps.append(  (colonne, float(sum(eap)))  )

    # Résultat final
    parentnode = sorted(eaps, key=lambda x: x[1])
    print(f"Racine de l'arbre de décision {'choisit par le programme' if raw else 'sélectionné à l\'aide de `scipy`'} : {parentnode[0]}")



def decisiontree(file: str, plot: bool = False) -> None:
    """
    Construit un arbre de décision à partir d'un fichier de données et l'affiche sous forme graphique ou textuelle.

    :param file: Nom du fichier
    :param plot: Si True, affiche l'arbre avec `matplotlib` ; sinon, affiche l'arbre sous forme textuelle.
    """

    df = pd.read_csv(file, sep=' ')
    
    # Numérisation de la colonne si nécessaire
    for colonne in df.columns:
        if df[colonne].dtype != 'int64':
            d = {elem: index for index, elem in enumerate(pd.unique(df[colonne]))}
            df[colonne] = df[colonne].map(d)
            
    train = df.iloc[:]

    # On sérpare la variables du résulats de la classification
    features = df.columns[:-1]
    classification = df.columns[-1]
    X = train[features]
    y = train[classification]

    # On entraine le model pour la création de l'arbre de décision selon le principe de minimisation de l'entropie
    dtree = DecisionTreeClassifier(criterion='entropy')
    dtree = dtree.fit(X, y)

    # Affichage de l'arbre créé
    if plot:
        plt.figure(figsize=(10,6))
        tree.plot_tree(dtree, feature_names=features, filled=True, rounded=True, fontsize=10)
        plt.show()
    else:
        tree_text = export_text(dtree, feature_names=list(X.columns))
        print(f"Arbre de décision choisit par la librairie `sklearn` : \n{tree_text}")



# --- main --------------------------------------------------------------------

import argparse

if __name__ == "__main__":

    # Gestion des parametres de l'invite de commande
    parser = argparse.ArgumentParser(description="Gestion des paramètres")
    parser.add_argument('--file', nargs='?', help="Fichier CSV en entrée", default='example.csv')
    parser.add_argument('--plot', help="Affichage de l'arbre de décision avec `matplotlib`", default='no')
    args = parser.parse_args()

    file = args.file
    display_tree = True if args.plot == 'yes' else False

    findparentnode(file)
    findparentnode(file, raw=False)
    decisiontree(file, plot=display_tree)

