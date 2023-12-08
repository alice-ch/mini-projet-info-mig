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

# import de dfc: compositions et dfD: densités

dfC = pd.read_csv('C:/Users/adelie.saule/MIG/hummingbird-master/hummingbird-master/hummingbird/dts/glass_mig_density_v2/CompoD23.csv', sep=' ')
dfD = pd.read_csv('C:/Users/adelie.saule/MIG/hummingbird-master/hummingbird-master/hummingbird/dts/glass_mig_density_v2/D_2.2_3.csv')


# calcul d'une nouvelle colonne masse molaire à partir de la composition

def massmolaire(name):
    return Formula(name).mass


elements = dfC.columns
print(massmolaire('SiO2'))


dfC['Sum'] =  sum(dfC[elem] for elem in dfC.columns)
dfC['M'] = sum(dfC[elem]*massmolaire(elem)/dfC['Sum'] for elem in elements)

dfC['volume_molaire'] = dfC['M']/dfD['Density at RT ( g/cm3 )']

# exclusion des volumes molaire aberrants
dfC = dfC[dfC['volume_molaire'] < 37]

#préparation des sous dataframe pour les régressions
x = dfC[elements]
y = dfC['volume_molaire']

 
# Regression linéaire du volume molaire
regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(x, y)
coeffs = regr.coef_
print(regr.intercept_)

modèle = {}
for i in range(len(elements)):
    modèle[elements[i]] = coeffs[i]
    
dfC['prediction'] = sum(dfC[elem]*modèle[elem] for elem in elements)


r_sqr = metrics.r2_score(dfC['volume_molaire'], dfC['prediction'])

plt.scatter(dfC['volume_molaire'], dfC['prediction'])
plt.title("Regression linéaire volume molaire,  R² =" + str(round(r_sqr, 3)))
plt.show()


# Polynomial regression



degree=1

modelVm = make_pipeline(PolynomialFeatures(degree),Ridge(alpha=1e-3))
modelVm.fit(x.to_numpy(),y.to_numpy())
y_plot = modelVm.predict(x.to_numpy())


plt.figure()
plt.plot(y.values,y_plot,'ko')
plt.plot([0.9*np.min(y),1.1*np.max(y)],
         [0.9*np.min(y),1.1*np.max(y)],'k')

r_sqr = metrics.r2_score(dfC['volume_molaire'], y_plot)
plt.title("Regression polynomiale de degré " + str(degree) + " du volume molaire,  R² =" + str(round(r_sqr, 3)))
plt.show()
    
print((modelVm.steps[1][1].intercept_))




df = pd.read_csv('C:/Users/adelie.saule/MIG/Database/YDboxydes.csv', sep = ' ')


elements2 = df.columns[:-2]

df["Young's Modulus at RT ( GPa )"].hist(bins = 50)

plt.show()

degree=2 
df['Sum'] =  sum(df[elem] for elem in elements2)
df['M'] = sum(df[elem]*massmolaire(elem)/df['Sum'] for elem in elements2)
df['volume_molaire']  = df['M']/df['Density at RT ( g/cm3 )']
df['EV'] = df['volume_molaire'] * df["Young's Modulus at RT ( GPa )"]

df['volume_molaire'].hist(bins = 50)
plt.show()

modelEVm = make_pipeline(PolynomialFeatures(degree),Ridge(alpha=1e-3))
modelEVm.fit(df[elements2].to_numpy(),df['EV'].to_numpy())
EVpredi = modelEVm.predict(df[elements2].to_numpy())

df['EVp'] = pd.Series(EVpredi)


r_sqr = metrics.r2_score(df['EV'], df['EVp'])
plt.scatter(df['EV'], df['EVp'])
plt.title("Regression polynomiale de degré " + str(degree) + " du produit E* Vm,  R² =" + str(round(r_sqr, 3)))
plt.show()

print("coefficients regression E*Vm et intercept" )
print((modelEVm.steps[1][1].coef_))
print((modelEVm.steps[1][1].intercept_))



# Etant donné une composition, calcul du module de Young prédit

data = [[ 60+i, 40-i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(41)]

df1 = pd.DataFrame(data = data, index = None, columns = elements)


dfex = df[df['SiO2'] + df['Na2O'] == 100][["Young's Modulus at RT ( GPa )", "Na2O"]]

        
def predictionE(df):
    elements = df.columns
    df['volume_molaire_predit'] = modelVm.predict(df.to_numpy())
    df['EVpredit'] = modelEVm.predict(df[elements].to_numpy())
    df['E'] = df['EVpredit']/df['volume_molaire_predit']
    plt.plot(df.Na2O.values, df.E.values)
    plt.grid()
    plt.title("Module de Young prédit pour un verre 100-x SiO2, x Na2O")
    plt.scatter(dfex['Na2O'], dfex["Young's Modulus at RT ( GPa )"])
    plt.show()
    
def predictionrot(df):
    elements = df.columns
    df['volume_molaire_predit'] = modelVm.predict(df.to_numpy())
    df['M'] = sum(df[elem]*massmolaire(elem)/100 for elem in elements)
    df['rot'] = df['M']/df['volume_molaire_predit']
    plt.plot(df.Na2O.values, df.rot.values)
    plt.title("rho prédit pour un verre 100-x SiO2, x Na2O")
    plt.grid()
    plt.show()

predictionE(df1)





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