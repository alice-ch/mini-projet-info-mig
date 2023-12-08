# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:40:07 2023

@author: raphael.poux
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from molmass import Formula
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


dfC = pd.read_csv('C:/Users/adelie.saule/MIG/hummingbird-master/hummingbird-master/hummingbird/dts/glass_mig_density_v2/CompoD23.csv', sep=' ')
dfD = pd.read_csv('C:/Users/adelie.saule/MIG/hummingbird-master/hummingbird-master/hummingbird/dts/glass_mig_density_v2/D_2.2_3.csv')



def massmolaire(name):
    return Formula(name).mass


elements = dfC.columns
print(len(elements))


dfC['Sum'] =  sum(dfC[elem] for elem in dfC.columns)
dfC['M'] = sum(dfC[elem]*massmolaire(elem)/dfC['Sum'] for elem in elements)

dfC['volume_molaire'] = dfC['M']/dfD['Density at RT ( g/cm3 )']

dfC = dfC[dfC['volume_molaire'] < 37]

x = dfC[elements]
y = dfC['volume_molaire']
 
# # with sklearn
regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(x, y)

coeffs = regr.coef_

modèle = {}
for i in range(len(elements)):
    modèle[elements[i]] = coeffs[i]
    
dfC['prediction'] = sum(dfC[elem]*modèle[elem] for elem in elements)

r_sqr = metrics.r2_score(dfC['volume_molaire'], dfC['prediction'])

plt.scatter(dfC['volume_molaire'], dfC['prediction'])
plt.title("R² =" + str(round(r_sqr, 3)))
plt.show()
# Polynomial regression
degree=3

model = make_pipeline(PolynomialFeatures(degree),Ridge(alpha=1e-3))
model.fit(x.to_numpy(),y.to_numpy())
y_plot = model.predict(x.to_numpy())


plt.figure()
plt.plot(y.values,y_plot,'ko')
plt.plot([0.9*np.min(y),1.1*np.max(y)],
         [0.9*np.min(y),1.1*np.max(y)],'k')

r_sqr = metrics.r2_score(dfC['volume_molaire'], y_plot)
plt.title("R² =" + str(round(r_sqr, 3)))
plt.show()

print(len(model.steps[1][1].coef_))
print((model.steps[1][1].intercept_))



# # -*- coding: utf-8 -*-
# """
# Created on Fri Dec  1 09:34:58 2023

# @author: adelie.saule
# """

# import pandas as pd
# from sklearn import linear_model


# df = pd.read_csv('database_2.csv', sep=' ')

# x = df[[" SiO2"," B2O3", " Al2O3", " MgO", " CaO", " BaO", " Li2O", " Na2O", " K2O", " Cu2O", " Rb2O", " Ag2O", " Cs2O", " Tl2O", " MnO", " FeO", " CoO", " NiO", " CuO", " ZnO", " SrO", " CdO", " PbO", " SnO"]]
# y = df[' Molar Volume at RT ( cm3/mol )']
 
# # # with sklearn
# regr = linear_model.LinearRegression(fit_intercept = False)
# regr.fit(x, y)

# coeffs = regr.coef_
# print(coeffs)