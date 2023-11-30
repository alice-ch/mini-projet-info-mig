import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/raphael.poux/Mig verre/Interglad_propre.csv")

# On garde que les verres

df = df[df['Glass No.'].str.contains('G')]
# 94 103 verres  
df.drop(['Organic Compound', 'Others', 'O'], axis = 1, inplace = True)

# On établit une liste des composants à laquelle on ajoute les propriétés physiques pour créer la liste des colonnes conservées
Liste_tous_composants = []
for colonne in df.columns:
    if "O" in colonne:
        Liste_tous_composants.append(colonne)

Colonnes_gardées = Liste_tous_composants[:]
Colonnes_gardées.append('Vickers Hardness 100g ( MPa )')
Colonnes_gardées.append("Young's Modulus at RT ( GPa )")
Colonnes_gardées.append('Density at RT ( g/cm3 )')
Colonnes_gardées.append('Fracture Toughness ( MPa.m1/2 )')

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

# selection d'oxydes dont la composition moyenne dans les oxydes est supérieure à p

def Ox_garde_Gp(df, p):
    D = df[Liste_tous_composants].mean().sort_values( ascending = False)
    return ([ (elem, D[elem]) for elem in D.index if D[elem] > p] , [ (elem, D[elem]) for elem in D.index if D[elem] <= p])

def supp(df, p):
    oxydes_sortis = [ elem[0] for elem in Ox_garde_Gp(df, p)[1]]
    oxydes_gardes = [ elem[0] for elem in Ox_garde_Gp(df, p)[0]]
    for x in oxydes_sortis:
      df.drop(x, axis = 1, inplace=True)
    df_composants = df[oxydes_gardes]
    df['Sum'] = df_composants.sum(axis=1)
    df = df[(df['Sum'] > 98) & (df['Sum'] <= 100)]
    return df[oxydes_gardes]

# Méthode alternative dont on est sûrs qu'elle ne fonctionne pas pour l'instant

"""def Ox_garde_Gr(df, r):
    D = df[Liste_tous_composants].mean()
    for elem in Liste_tous_composants:
        D[elem] = df[elem][df[elem] > 0].count()
    return [ (elem, D[elem]/len(df)*100) for elem in D.sort_values(ascending = False).index if D[elem]/len(df)*100 > r]

def supr(df, p):
    oxydes_sortis = [ elem[0] for elem in Ox_garde_Gr(df, p)[1]]
    oxydes_gardes = [ elem[0] for elem in Ox_garde_Gr(df, p)[0]]
    for x in oxydes_sortis:
      df.drop(x, axis = 1, inplace=True)
    df_composants = df[[nom[0] for nom in oxydes_gardes]]
    df['Sum'] = df_composants.sum(axis=1)
    df = df[(df['Sum'] > 98) & (df['Sum'] <= 100)]
    return df[oxydes_gardes]"""

def Gp2(df, p):
    D = supp(df, p).mean().sort_values( ascending = False)
    return [ elem for elem in D.index if D[elem] > p]

# Le nom des variables se raccourcit car on commence à avoir la flemme

# On reforme une liste de colonnes gardées avec que les oxydes gardés

oxgard = Gp2(df, p)
cogardees = oxgard[:]
cogardees.append('Vickers Hardness 100g ( MPa )')
cogardees.append("Young's Modulus at RT ( GPa )")
cogardees.append('Density at RT ( g/cm3 )')
cogardees.append('Fracture Toughness ( MPa.m1/2 )')

# On fait le tri des valeurs aberrantes

df = df[cogardees]

df['Vickers Hardness 100g_check ( MPa )'] = (df["Vickers Hardness 100g ( MPa )"] > 3000) & (df["Vickers Hardness 100g ( MPa )"] <= 7500)
dfV = df.loc[df['Vickers Hardness 100g_check ( MPa )'] == True]

df['Young_check at RT ( GPa )'] = (df["Young's Modulus at RT ( GPa )"] > 50) & (df["Young's Modulus at RT ( GPa )"] <= 130)
dfY = df.loc[df['Young_check at RT ( GPa )'] == True]

df['Density at RT_check ( g/cm3 )'] = (df["Density at RT ( g/cm3 )"] > 2) & (df["Density at RT ( g/cm3 )"] <= 4)
dfD = df.loc[df['Density at RT_check ( g/cm3 )'] == True]

df['Fracture Toughness_check ( MPa.m1/2 )'] = (df["Fracture Toughness ( MPa.m1/2 )"] > 0.5) & (df["Fracture Toughness ( MPa.m1/2 )"] <= 1.5)
dfT = df.loc[df['Fracture Toughness_check ( MPa.m1/2 )'] == True]

# On écrit les dataframes dans des fichiers

dfT[oxgard].to_csv('compoT.csv', sep = ' ', index=False)
dfT["Fracture Toughness ( MPa.m1/2 )"].to_csv('T.csv', sep = ' ', index=False)
dfD[oxgard].to_csv('CompoD.csv', sep = ' ', index=False)
dfD["Density at RT ( g/cm3 )"].to_csv('D.csv', sep = ' ', index=False)
dfV[oxgard].to_csv('CompoV.csv', sep = ' ', index=False)
dfV["Vickers Hardness 100g ( MPa )"].to_csv('V.csv', sep = ' ', index=False)
dfY[oxgard].to_csv('CompoY.csv', sep = ' ', index=False)
dfY["Young's Modulus at RT ( GPa )"].to_csv('Y.csv', sep = ' ', index=False)
