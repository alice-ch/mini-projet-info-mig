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
