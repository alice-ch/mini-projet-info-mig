import numpy as np
import pandas as pd
from math import *

df = pd.read_csv('D:/Users/Valentin/Desktop/1A Mines/Semestre 1/UE15 - MIG/mini-projet-info-mig/Dataframe_test.csv')

# On vire les colonnes pas utiles
liste_col_a_enlever = ['No.', ' Glass No.', ' Data Source', ' Year', ' Data Source Number']

df.drop(liste_col_a_enlever, axis = 1, inplace=True)        # Enlève les colonnes inutiles

# On convertit tout en float
df = df.replace(' *', np.nan)               # * -> NaN
df = df.replace(' ', np.nan)                # ' ' -> NaN
df = df.fillna(0)                           # NaN -> 0

df = df.astype(float)                       # type object -> type float

Liste_tous_composants = [' SiO2', ' B2O3', ' Al2O3', ' CaO', ' Na2O']


# tri de la liste pour avoir une liste de couples (oxydes, pourcentages de présence) avec un pourcentage de présence décroissant

def fusion(liste1,liste2):
    liste=[]
    i,j=0,0
    while i<len(liste1)and j<len(liste2):
        if liste1[i][1]>=liste2[j][1]:
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

# On drop les duplicatas

df.drop_duplicates(keep='first', inplace=True)                         # Les duplicatas parfaits
df.drop_duplicates(subset=Liste_tous_composants, keep=False, inplace=True)       # Les compos identiques

# selection d'oxydes
Liste_oxyde_proportion = []
n = len(df)
for x in Liste_tous_composants:
    Liste_oxyde_proportion.append( (x, df[x][df[x] > 0.1].count()/n) )
print(tri_fusion(Liste_oxyde_proportion))

def coupe_liste_triee(L, proportion_apparition_minimum):
    Oxydes_gardes = []
    Oxydes_sortis = []
    for i in range(len(L)):
        if L[i][1] < proportion_apparition_minimum:
            Oxydes_gardes = L[:i]
            Oxydes_sortis = L[i:]
            return Oxydes_gardes, Oxydes_sortis
    return L, []

Oxydes_sortis = coupe_liste_triee(tri_fusion(Liste_oxyde_proportion), 0.1)[1]

for (x,i) in Oxydes_sortis:
    df.drop(x, axis = 1, inplace=True)

#liste d'oxydes de Raviner 2020 pour comparaison
#  SiO2, B2O3, Al2O3, MgO, CaO, BaO, Li2O,
# Na2O, K2O, Ag2O, Cs2O, Tl2O, BeO, NiO, CuO, ZnO, CdO, PbO,
# Ga2O3, Y2O3, La2O3, Gd2O3, Bi2O3, TiO2, ZrO2, TeO2, P2O5, V2O5,
# Nb2O5, Ta2O5, MoO3, WO3, H2O, Sm2O3, MgF2, PbF2, PbCl2

df['sum_check'] = (df['Sum'] > 98) & (df['Sum'] <= 100)

df = df.loc[df['sum_check'] == True]

print(df.head(5))