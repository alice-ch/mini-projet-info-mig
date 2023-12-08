# Generic imports
import math
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
from hummingbird.app.base_app        import *
from hummingbird.src.dataset.dataset import *
from hummingbird.src.agent.nn        import *

nbr_oxydes = 11

X = np.linspace(0, 50, 100)
X_net = [[100 - x, x, 0, 0, 0, 0, 0, 0, 0, 0, 0] for x in X]

mean = np.mean(X_net)       # On normalise les entrées
std = np.std(X_net)
X_net = (X_net - mean)/std
X_net = torch.Tensor(X_net)


net = nn(pms(model=pms(inp_dim=nbr_oxydes,
                       arch=[200,200,200,1],
                       acts=["relu","relu","relu","linear"]),
             loss="mse",
             lr=1.0e-3))

net.load("C:/Users/adelie.saule/MIG/Courbe_évolution_densité_SiO2_Na2O/net_density.txt")

Y = list(net.apply(X_net))

plt.plot(X, Y)
plt.grid()
plt.title("Composition: 100-x% SiO2, x% Na2O")
plt.ylabel("Densité en g/cm3")
plt.xlabel("x")
plt.show()


