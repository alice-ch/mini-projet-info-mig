# Generic imports
import math
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Custom imports
from hummingbird.app.base_app        import *
from hummingbird.src.dataset.dataset import *
from hummingbird.src.agent.nn        import *

nbr_oxydes = 13

X = np.linspace(0, 40, 100)
X_net = np.array([[100 - x, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0] for x in X])
mean = np.mean(X_net, axis=0)       # On normalise les entrées
std = np.std(X_net, axis=0)
X_net = (X_net - mean)/std
X_net[np.isnan(X_net)] = 0          # Remplace les 0/0 = NaN par des 0
X_net = torch.Tensor(X_net)

net = nn(pms(model=pms(inp_dim=nbr_oxydes,
                       arch=[500,500,100,50,1],
                       acts=["relu","relu","relu","relu","linear"]),
             loss="mse",
             lr=1.0e-4))

net.load("C:/Users/adelie.saule/MIG/Courbe_évolution_densité_SiO2_Na2O/net_density.txt")

Y = list(net.apply(X_net))

df = pd.read_csv("//filer-2/Students/adelie.saule/Bureau/Résultats ML/Résultats densité/Dataset vendredi/CompoD.csv", sep=" ")
labels = pd.read_csv("//filer-2/Students/adelie.saule/Bureau/Résultats ML/Résultats densité/Dataset vendredi/D.csv", sep=" ")

df['Sum'] = df[["SiO2", "K2O"]].sum(axis=1)

df['sum_check'] = (df['Sum'] == 100)

df = df.loc[df['sum_check'] == True]            # On garde que ce qui est composé de SiO2 et Na2O

df = df.loc[df["K2O"] <= 40]                   # On garde que <40% de Na2O sinon ça a pas de sens

ind = df.index.values.astype(int)

labels = list(labels.iloc[ind]["Density at RT ( g/cm3 )"])

X2 = list(df['K2O'])

plt.scatter(X2, labels)
plt.plot(X, Y, c='r')
plt.grid()
plt.title("Composition: 100-x% SiO2, x% K2O")
plt.ylabel("Densité en g/cm3")
plt.xlabel("x")
plt.show()