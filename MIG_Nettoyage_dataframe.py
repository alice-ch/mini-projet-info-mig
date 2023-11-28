import pandas as pd
from math import *
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('D:/Users/Valentin/Desktop/1A Mines/Semestre 1/UE15 - MIG/mini-projet-info-mig/Dataframe_test.csv')

# On vire les colonnes pas utiles
liste_col_a_enlever = ['No.', ' Glass No.', ' Data Source', ' Year', ' Data Source Number']
# Liste des oxydes trouvés par Selection_oxydes
liste_composants = [' SiO2', ' B2O3', ' Al2O3', ' CaO', ' Na2O']

df.drop(liste_col_a_enlever, axis = 1, inplace=True)        # Enlève les colonnes inutiles

# On convertit tout en float
df = df.replace(' *', np.nan)               # * -> NaN
df = df.replace(' ', np.nan)                # ' ' -> NaN
df = df.fillna(0)                           # NaN -> 0

df = df.astype(float)                       # type object -> type float

#On crée une colonne 'Somme'
df_composants = df[liste_composants]
df['Sum'] = df_composants.sum(axis=1)       # Crée une nouvelle colonne pour connaitre la somme des % de composition


def garde_Young(df):
    """
    Méthode qui supprime les lignes où le module d'Young à temp. ambiante n'est pas renseigné
    """
    df['Young_check'] = (df["Young's modulus"] > 98) & (df["Young's modulus"] <= 100)
    df = df.loc[df['Young_check'] == True]

def garde_densité(df):
    """
    Méthode qui supprime les lignes où la densité à temp. ambiante n'est pas renseignée
    """
    df = df.loc[df[' Density at RT ( g/cm3 )'] != 0]

df['sum_check'] = (df['Sum'] > 98) & (df['Sum'] <= 100)

df = df.loc[df['sum_check'] == True]
