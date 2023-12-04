# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 14:40:07 2023

@author: raphael.poux
"""


import pandas as pd
from sklearn import linear_model

dfC = pd.read_csv('C:/Users/raphael.poux/CompoD23.csv', sep=' ')
dfD = pd.read_csv('C:/Users/raphael.poux/D23.csv', sep=' ')

dico = {'SiO2':60.08, 'Na2O':61.9789, 'B2O3':69.6182, 'CaO':56.0774, 'Al2O3':101.96, 'K2O':94.2, 'Li2O':29.881, 'MgO':40.3044, 'PbO':223.2, 'BaO':153.33, 'TiO2':79.866, 'GeO2':104.61, 'ZnO':81.38}

def massmolaire(nom):
    return dico[nom]
    
elements = dfC.columns

massmolaire('SiO2')

dfC['Sum'] =  sum(dfC[elem] for elem in dfC.columns)
dfC['M'] = sum(dfC[elem]*massmolaire(elem)/dfC['Sum'] for elem in elements)

dfD['volume_molaire'] = dfC['M']/dfD['Density at RT ( g/cm3 )']



x = dfC[elements]
y = dfD['volume_molaire']
 
# # with sklearn
regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(x, y)

coeffs = regr.coef_
print(coeffs)


# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:34:58 2023

@author: adelie.saule
"""

import pandas as pd
from sklearn import linear_model


df = pd.read_csv('database_2.csv', sep=' ')

x = df[[" SiO2"," B2O3", " Al2O3", " MgO", " CaO", " BaO", " Li2O", " Na2O", " K2O", " Cu2O", " Rb2O", " Ag2O", " Cs2O", " Tl2O", " MnO", " FeO", " CoO", " NiO", " CuO", " ZnO", " SrO", " CdO", " PbO", " SnO"]]
y = df[' Molar Volume at RT ( cm3/mol )']
 
# # with sklearn
regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(x, y)

coeffs = regr.coef_
print(coeffs)