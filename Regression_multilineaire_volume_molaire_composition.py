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

def predictionv(df):
    #but: rajouter une colonne qui rajoute le volume molaire prédit calculé à partir des masses 