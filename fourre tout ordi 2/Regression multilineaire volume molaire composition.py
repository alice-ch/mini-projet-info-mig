# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:34:58 2023

@author: adelie.saule
"""

import pandas as pd
from sklearn import linear_model


df = pd.read_csv('volume_molaire_composition_test')

x = df[['interest_rate','unemployment_rate']]
y = df['index_price']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)
