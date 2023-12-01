# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:47:14 2023

@author: adelie.saule
"""

import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/adelie.saule/MIG/Regression/database_1.csv", header = 25)

#dans database_1 il y a des espaces avant tous les oxydes et le header est plus bas
print(df.columns)

# On établit une liste des composants à laquelle on ajoute le volume molaire et les index des verres pour créer la liste des colonnes conservées
Liste_tous_composants = []
for colonne in df.columns:
    if "O" in colonne:
        Liste_tous_composants.append(colonne)

Colonnes_gardées = Liste_tous_composants[:]
Colonnes_gardées.append(' Molar Volume at RT ( cm3/mol )')
# Colonnes_gardées.append(' Glass No.')
print(Colonnes_gardées)

df = df[Colonnes_gardées]

# On convertit tout en float

df = df.replace(' *', np.nan)               # * -> NaN
df = df.replace(' ', np.nan)                # ' ' -> NaN
df = df.fillna(0)                           # NaN -> 0

for i in df.columns:
    df[i] = pd.to_numeric(df[i], errors='coerce')                       # type object -> type float

df = df.fillna(0)                           # NaN -> 0



# On drop les duplicatas

print('avant duplicata' + str(len(df)))

df.drop_duplicates(keep='first', inplace=True)

print('après duplicata strictement identiques' + str(len(df)))                         # Les duplicatas parfaits

df.drop_duplicates(subset=Liste_tous_composants, keep=False, inplace=True)       # Les compos identiques    inutile a priori

print('après duplicata de composition' + str(len(df)))


# On écrit les dataframes dans des fichiers

df.to_csv('database_2.csv', sep = ' ', index=False)

