import pandas as pd
import numpy as np
from math import *

df = pd.read_csv("C:/Users/raphael.poux/Mig verre/Interglad_propre.csv")

# On garde que les verres

df = df[df['Glass No.'].str.contains('G')]

# On vire les colonnes pas utiles
liste_col_a_enlever = [' Unnamed: 0', ' No.', ' Glass No.', ' Data Source', ' Year', ' Data Source Number', ' Thermal Treatment',
                       ' Bulk Density ( g/cm3 )', ' Density (Miscell) ( g/cm3 )',  " Young's Modulus (Miscell) ( GPa )", " Young's Modulus at <0C ( GPa )",
                       ' Vickers Hardness 50g ( MPa )', ' Vickers Hardness (Typical) ( MPa )', ' Vickers Hardness 200g ( MPa )', ' Vickers Hardness 500g ( MPa )', ' Vickers Hardness (Miscell) ( MPa )',
                       ' H', ' Li', ' B', ' C', ' N', ' O', ' F', ' Na', ' Mg', ' Al', ' Si', ' P', ' S', ' Cl', ' K', ' Ca', ' Sc', ' Ti', ' V', ' Cr', ' Mn', ' Fe', ' Co', ' Ni', ' Cu', ' Zn', ' Ga', ' Ge', ' As', ' Se', ' Br', ' Rb', ' Y', ' Zr', ' Nb', ' Mo', ' Pd', ' Ag', ' Cd', ' In', ' Sn', ' Sb', ' Te', ' I', ' Ba', ' La', ' Ce', ' Gd', ' W', ' Re', ' Pt', ' Au', ' Hg', ' Tl', ' Pb', ' Bi', ' Th', ' U',
                       ]

liste_enlever = [ elem[1:] for elem in liste_col_a_enlever]

df.drop(liste_enlever, axis = 1, inplace=True)        # Enlève les colonnes inutiles


# On convertit tout en float
df = df.replace(' *', np.nan)               # * -> NaN
df = df.replace(' ', np.nan)                # ' ' -> NaN
df = df.fillna(0)                           # NaN -> 0

for i in df.columns:
    df[i] = pd.to_numeric(df[i], errors='coerce')                       # type object -> type float

df = df.fillna(0)                           # NaN -> 0
df = df[df['SiO2'] > 0]                     # on ne garde que les verres silicatés
print(df.head(5))

Liste_tous_composants = []
for colonne in df.columns:
    if "O" in colonne:
        Liste_tous_composants.append(colonne)

Liste_tous_composants.remove('O')
Liste_tous_composants.remove('Organic Compound')
Liste_tous_composants.remove('Others')

# On drop les duplicatas

df.drop_duplicates(keep='first', inplace=True)                         # Les duplicatas parfaits
df.drop_duplicates(subset=Liste_tous_composants, keep=False, inplace=True)       # Les compos identiques

# selection d'oxydes

def Ox_garde_Gp(df, p):
    D = df[Liste_tous_composants].mean().sort_values( ascending = False)
    return ([ (elem, D[elem]) for elem in D.index if D[elem] > p] , [ (elem, D[elem]) for elem in D.index if D[elem] <= p])

def Ox_garde_Gr(df, r):
    D = df[Liste_tous_composants].mean()
    for elem in Liste_tous_composants:
        D[elem] = df[elem][df[elem] > 0].count()
    return [ (elem, D[elem]/len(df)*100) for elem in D.sort_values(ascending = False).index if D[elem]/len(df)*100 > r]



#liste d'oxydes de Raviner 2020 pour comparaison
# SiO2, B2O3, Al2O3, MgO, CaO, BaO, Li2O,
# Na2O, K2O, Ag2O, Cs2O, Tl2O, BeO, NiO, CuO, ZnO, CdO, PbO,
# Ga2O3, Y2O3, La2O3, Gd2O3, Bi2O3, TiO2, ZrO2, TeO2, P2O5, V2O5,
# Nb2O5, Ta2O5, MoO3, WO3, H2O, Sm2O3, MgF2, PbF2, PbCl2

#On crée une colonne 'Somme'
df_composants = df[[nom[0] for nom in Oxydes_gardes]]
df['Sum'] = df_composants.sum(axis=1)       # Crée une nouvelle colonne pour connaitre la somme des % de composition

df['sum_check'] = (df['Sum'] > 98) & (df['Sum'] <= 100)      # On ne garde que les 100% composition
df = df.loc[df['sum_check'] == True]

def garde_Young(df, inf, sup):
    """
    Méthode qui supprime les lignes où le module d'Young à temp. ambiante n'est pas renseigné
    On peut choisir 50 et 130 pour les bornes
    """
    df['Young_check at RT ( GPa )'] = (df["Young's modulus at RT ( GPa )"] > inf) & (df["Young's modulus at RT ( GPa )"] <= sup)
    df = df.loc[df['Young_check at RT ( GPa )'] == True]

def garde_Vickers(df, inf, sup):
    """
    Méthode qui supprime les lignes où le module d'Young à temp. ambiante n'est pas renseigné
    On peut choisir 3000 et 7500 pour les bornes
    """
    df['Vickers Hardness 100g_check ( MPa )'] = (df["Vickers Hardness 100g ( MPa )"] > inf) & (df["Vickers Hardness 100g ( MPa )"] <= sup)
    df = df.loc[df['Vickers Hardness 100g_check ( MPa )'] == True]

def garde_densite(df, inf, sup):
    """
    Méthode qui supprime les lignes où le module d'Young à temp. ambiante n'est pas renseigné
    On peut choisir 2 et 4 pour les bornes
    """
    df['Density at RT_check ( g/cm3 )'] = (df["Density at RT ( g/cm3 )"] > inf) & (df["Density at RT ( g/cm3 )"] <= sup)
    df = df.loc[df['Density at RT_check ( g/cm3 )  '] == True]

def garde_toughness(df, inf, sup):
    """
    Méthode qui supprime les lignes où le module d'Young à temp. ambiante n'est pas renseigné
    On peut choisir 0.5 et 1.5 pour les bornes
    """
    df['Fracture Toughness_check ( MPa.m1/2 )'] = (df["Fracture Toughness ( MPa.m1/2 )"] > inf) & (df["Fracture Toughness ( MPa.m1/2 )"] <= sup)
    df = df.loc[df['Fracture Toughness_check ( MPa.m1/2 )'] == True]

df.head()




