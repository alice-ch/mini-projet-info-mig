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

nbr_oxydes = 11
#SiO2 Al2O3 CaO Na2O MgO B2O3 Li2O GeO2 K2O P2O5 BaO
X_net = np.array([69.11, 3.3, 7.4, 14.99, 5.0, 0, 0, 0, 0, 0, 0])
mean = np.mean(X_net, axis=0)       # On normalise les entrées
std = np.std(X_net, axis=0)
X_net = (X_net - mean)/std
X_net[np.isnan(X_net)] = 0          # Remplace les 0/0 = NaN par des 0
X_net = torch.Tensor(X_net)

net = nn(pms(model=pms(inp_dim=nbr_oxydes,
                       arch=[1000,500,100,1],
                       acts=["relu","relu","relu","linear"]),
             loss="mse",
             lr=1.0e-3))

net.load("C:/Users/adelie.saule/MIG/Courbe_évolution_densité_SiO2_Na2O/net_young.txt")

Y = list(net.apply(X_net))

print(Y)