import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('D:/Users/Valentin/Desktop/1A Mines/Semestre 1/UE15 - MIG/Dataframe_test.csv', index_col='No.')

liste_col_a_enlever = [' Glass No.', ' Data Source', ' Year', ' Data Source Number']
liste_composants = [' SiO2', ' B2O3', ' Al2O3', ' CaO', ' Na2O']

df.drop(liste_col_a_enlever, axis = 1, inplace=True)        # Enlève les colonnes inutiles

pd.to_numeric(df, )

df_composants = df[liste_composants]
df['Sum'] = df_composants.sum(axis=1)                       # Crée une nouvelle colonne pour connaitre la somme des % de composition

print(df.head(5))

# selection d'oxydes
L = []
n = len(df)
for x in df.columns[0]:
    L.append( (x, df[x][df[x] > 0.1].count()/n) )

def coupe_liste_triee(L, a):
    Oxydes_gardes = []
    Oxydes_sortis = []
    for i in range(len(L)):
        if L[i][1] < a:
            Oxydes_gardes = L[:i]
            Oxydes_sortis = L[i:]
            return Oxydes_gardes, Oxydes_sortis

#liste d'oxydes de Raviner 2020 pour comparaison
#  SiO2, B2O3, Al2O3, MgO, CaO, BaO, Li2O,
# Na2O, K2O, Ag2O, Cs2O, Tl2O, BeO, NiO, CuO, ZnO, CdO, PbO,
# Ga2O3, Y2O3, La2O3, Gd2O3, Bi2O3, TiO2, ZrO2, TeO2, P2O5, V2O5,
# Nb2O5, Ta2O5, MoO3, WO3, H2O, Sm2O3, MgF2, PbF2, PbCl2

for (x,i) in Oxydes_sortis[]
    df.drop(x, axis = 1, inplace=True)

df['sum_check'] = np.where(98 < df['sum'] < 100, True, False)

df.drop(df[df['sum_check']==False].index, inplace=True)
