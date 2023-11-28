import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('D:/Users/Valentin/Desktop/1A Mines/Semestre 1/UE15 - MIG/mini-projet-info-mig/Dataframe_test.csv', index_col='No.')

liste_col_a_enlever = [' Glass No.', ' Data Source', ' Year', ' Data Source Number']
liste_composants = [' SiO2', ' B2O3', ' Al2O3', ' CaO', ' Na2O']

df.drop(liste_col_a_enlever, axis = 1, inplace=True)        # Enlève les colonnes inutiles

df = df.replace(' *', np.nan)               # * -> NaN
df = df.replace(' ', np.nan)                # ' ' -> NaN
df = df.fillna(0)                           # NaN -> 0

df = df.astype(float)                       # type object -> type float

df_composants = df[liste_composants]
df['Sum'] = df_composants.sum(axis=1)       # Crée une nouvelle colonne pour connaitre la somme des % de composition


def garde_Young(df):
    """
    Méthode qui supprime les lignes où le module d'Young à temp. ambiante n'est pas renseigné
    """
    df.dropna(subset=[" Young's Modulus at RT ( GPa )"])

def garde_densité(df):
    """
    Méthode qui supprime les lignes où la densité à temp. ambiante n'est pas renseignée
    """
    df.dropna(subset=[' Density at RT ( g/cm3 )'])