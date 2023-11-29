import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/raphael.poux/Mig verre/Interglad_propre.csv")

# On garde que les verres

df = df[df['Glass No.'].str.contains('G')]
df.drop(['Organic Compound', 'Others', 'O'], axis = 1, inplace = True)


Liste_tous_composants = []
for colonne in df.columns:
    if "O" in colonne:
        Liste_tous_composants.append(colonne)

Colonnes_gardées = Liste_tous_composants[:]
Colonnes_gardées.append('Vickers Hardness 100g ( MPa )')
Colonnes_gardées.append("Young's Modulus at RT ( GPa )")
Colonnes_gardées.append('Density at RT ( g/cm3 )')
Colonnes_gardées.append('Fracture Toughness ( MPa.m1/2 )')

#print(Colonnes_gardées)

df = df[Colonnes_gardées]

# On convertit tout en float
df = df.replace(' *', np.nan)               # * -> NaN
df = df.replace(' ', np.nan)                # ' ' -> NaN
df = df.fillna(0)                           # NaN -> 0

for i in df.columns:
    df[i] = pd.to_numeric(df[i], errors='coerce')                       # type object -> type float

df = df.fillna(0)                           # NaN -> 0
df = df[df['SiO2'] > 0]                     # on ne garde que les verres silicatés



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


def supp(df, p):
    oxydes_sortis = [ elem[0] for elem in Ox_garde_Gp(df, p)[1]]
    oxydes_gardes = [ elem[0] for elem in Ox_garde_Gp(df, p)[0]]
    print(oxydes_gardes)
    for x in oxydes_sortis:
      df.drop(x, axis = 1, inplace=True)
    df_composants = df[oxydes_gardes]
    df['Sum'] = df_composants.sum(axis=1)
    df = df[(df['Sum'] > 98) & (df['Sum'] <= 100)]
    return df[oxydes_gardes]

def supr(df, p):
    oxydes_sortis = [ elem[0] for elem in Ox_garde_Gr(df, p)[1]]
    oxydes_gardes = [ elem[0] for elem in Ox_garde_Gr(df, p)[0]]
    for x in oxydes_sortis:
      df.drop(x, axis = 1, inplace=True)
    df_composants = df[[nom[0] for nom in oxydes_gardes]]
    df['Sum'] = df_composants.sum(axis=1)
    df = df[(df['Sum'] > 98) & (df['Sum'] <= 100)]
    return df[oxydes_gardes]

def Gp2(df, p):
    D = supp(df, p).mean().sort_values( ascending = False)
    return [ (elem, D[elem]) for elem in D.index if D[elem] > p]

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
    df = df.loc[df['Density at RT_check ( g/cm3 )'] == True]

def garde_toughness(df, inf, sup):
    """
    Méthode qui supprime les lignes où le module d'Young à temp. ambiante n'est pas renseigné
    On peut choisir 0.5 et 1.5 pour les bornes
    """
    df['Fracture Toughness_check ( MPa.m1/2 )'] = (df["Fracture Toughness ( MPa.m1/2 )"] > inf) & (df["Fracture Toughness ( MPa.m1/2 )"] <= sup)
    df = df.loc[df['Fracture Toughness_check ( MPa.m1/2 )'] == True]

df.head()




