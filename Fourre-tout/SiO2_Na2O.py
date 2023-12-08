import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("//filer-2/Students/adelie.saule/Bureau/Résultats ML/Résultats densité/Dataset vendredi/CompoD.csv", sep=" ")
labels = pd.read_csv("//filer-2/Students/adelie.saule/Bureau/Résultats ML/Résultats densité/Dataset vendredi/D.csv", sep=" ")

df['Sum'] = df[["SiO2", "Na2O"]].sum(axis=1)

df['sum_check'] = (df['Sum'] == 100)

df = df.loc[df['sum_check'] == True]

ind = df.index.values.astype(int)

labels = list(labels.iloc[ind]["Density at RT ( g/cm3 )"])

X = list(df['Na2O'])

x, y = np.meshgrid(X, labels)

plt.scatter(X, labels)
plt.grid()
plt.title("Composition: 100-x% SiO2, x% Na2O")
plt.ylabel("Densité en g/cm3")
plt.xlabel("x")
plt.show()