import pandas as pd
from math import *
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

print(df.head(5))

# selection d'oxydes
L = []
n = len(df)
for x in df.columns[0]:
    L.append( (x, df[x][df[x] > 0.1].count()/n) )




# tri de la liste pour avoir une liste de couples (oxydes, pourcentages de présence) avec un pourcentage de présence décroissant


def fusion(liste1,liste2):
    liste=[]
    i,j=0,0
    while i<len(liste1)and j<len(liste2):
        if liste1[i][1]<=liste2[j][1]:
            liste.append(liste1[i])
            i+=1
        else:
            liste.append(liste2[j])
            j+=1
    while i<len(liste1):
        liste.append(liste1[i])
        i+=1
    while j<len(liste2):
        liste.append(liste2[j])
        j+=1
    return liste


def tri_fusion(liste):
    if len(liste)<2:
        return liste[:]
    else:
        milieu=len(liste)//2
        liste1=tri_fusion(liste[:milieu])
        liste2=tri_fusion(liste[milieu:])
        return fusion(liste1,liste2)
