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
import math as m

# import de dfc: compositions et dfD: densités

dfC = pd.read_csv('C:/Users/adelie.saule/MIG/hummingbird-master/hummingbird-master/hummingbird/dts/glass_mig_density_v2/CompoD23.csv', sep=' ')
dfD = pd.read_csv('C:/Users/adelie.saule/MIG/hummingbird-master/hummingbird-master/hummingbird/dts/glass_mig_density_v2/D_2.2_3.csv')

# calcul d'une nouvelle colonne masse molaire à partir de la composition

def massmolaire(name):
    return Formula(name).mass

elements = dfC.columns

dfC['Sum'] =  sum(dfC[elem] for elem in dfC.columns)
dfC['M'] = sum(dfC[elem]*massmolaire(elem)/dfC['Sum'] for elem in elements)

# on en déduit le volume molaire

dfC['volume_molaire'] = dfC['M']/dfD['Density at RT ( g/cm3 )']

# exclusion des volumes molaire aberrants
dfC = dfC[dfC['volume_molaire'] < 37]

#préparation des sous dataframe pour les régressions qui donnent le volume molaire
x = dfC[elements]
y = dfC['volume_molaire']

 
# Regression linéaire du volume molaire
regr = linear_model.LinearRegression(fit_intercept = False)
regr.fit(x, y)

coeffs = regr.coef_
intercept = regr.intercept_

print('coefficients et intercept regression linéaire volume molaire')
print(elements)
print(coeffs)
print('intercept:')
print(intercept)

#enregistrement de la regression et prédiction
modèle = {}
for i in range(len(elements)):
    modèle[elements[i]] = coeffs[i]
    
dfC['prediction'] = sum(dfC[elem]*modèle[elem] for elem in elements)


r_sqr = metrics.r2_score(dfC['volume_molaire'], dfC['prediction'])

#tracé de la régression linéaire
plt.scatter(dfC['volume_molaire'], dfC['prediction'])
plt.xlabel(xlabel = 'Volume molaire expérimental')
plt.ylabel(ylabel = 'Volume molaire prédit')
plt.title("Regression linéaire volume molaire,  R² =" + str(round(r_sqr, 3)))
plt.savefig('C:/Users/adelie.saule/MIG/Resultats_de_regressions/Regression_lineaire_volume_molaire.png')



# Regression polynomiale pour le volume molaire
degree=1

modelVm = make_pipeline(PolynomialFeatures(degree),Ridge(alpha=1e-3))
modelVm.fit(x.to_numpy(),y.to_numpy())
y_plot = modelVm.predict(x.to_numpy())

#tracé
plt.figure()
plt.plot(y.values,y_plot,'ko')
plt.xlabel(xlabel = 'Volume molaire (cm3/mol)')
plt.ylabel(ylabel = 'Ecart relatif (%)')
plt.plot([0.9*np.min(y),1.1*np.max(y)],
         [0.9*np.min(y),1.1*np.max(y)],'k')

r_sqr = metrics.r2_score(dfC['volume_molaire'], y_plot)
plt.title("Regression polynomiale de degré " + str(degree) + " du volume molaire,  R² =" + str(round(r_sqr, 3)))
plt.show()

dfC['ecart_relatif_V'] = (y-pd.Series(y_plot))/y*100
plt.scatter(y.values, dfC['ecart_relatif_V'].values)
plt.title("Ecart relatif volume molaire")
plt.xlabel(xlabel = 'Volume molaire (cm3/mol)')
plt.ylabel(ylabel = 'Ecart relatif (%)')
plt.savefig('C:/Users/adelie.saule/MIG/Resultats_de_regressions/Ecart_relatif_volume_molaire.png')
plt.show()

#extraction de la constante    
print((modelVm.steps[1][1].intercept_))



# même modèle pour EV: regression de degré 2
df = pd.read_csv('C:/Users/adelie.saule/MIG/Database/YDboxydes.csv', sep = ' ')


elements2 = df.columns[:-2]
#observer la répartition des données de E
df["Young's Modulus at RT ( GPa )"].hist(bins = 50)
plt.title('Repartition E dans base')
plt.savefig('C:/Users/adelie.saule/MIG/Resultats_de_regressions/Repartition_E_dans_base.png')
plt.show()

#on génère la colonne EV expérimentale
df['Sum'] =  sum(df[elem] for elem in elements2)
df['M'] = sum(df[elem]*massmolaire(elem)/df['Sum'] for elem in elements2)
df['volume_molaire']  = df['M']/df['Density at RT ( g/cm3 )']
df['EV'] = df['volume_molaire'] * df["Young's Modulus at RT ( GPa )"]


#on observe la répartition des données de volume molaire
df['volume_molaire'].hist(bins = 50)
plt.title('Repartition Vm dans base')
plt.savefig('C:/Users/adelie.saule/MIG/Resultats_de_regressions/Repartition_Vm_dans_base.png')
plt.show()

#regression polynomiale pour déterminer EV
degree=2 
modelEVm = make_pipeline(PolynomialFeatures(degree),Ridge(alpha=1e-3))
modelEVm.fit(df[elements2].to_numpy(),df['EV'].to_numpy())
EVpredi = modelEVm.predict(df[elements2].to_numpy())
df['EVp'] = pd.Series(EVpredi)

df['ecart_relatif_Ev'] = (df['EV']-df["EVp"])/df["EV"]*100
plt.scatter(df.EV.values, df['ecart_relatif_Ev'].values)
plt.title("Ecart relatif EVmol")
plt.xlabel(xlabel = "EVmol (GPa.cm3/mol)")
plt.ylabel(ylabel = 'Ecart relatif (%)')
plt.show()

plt.scatter(df.EV.values, df['EVp'].values)
plt.title("Prédiction de Evmol")
plt.xlabel(xlabel = 'Valeurs expérimentales de EVmol ((GPa.cm3/mol))')
plt.ylabel(ylabel = 'Valeurs prédites de EVmol (GPa.cm3/mol)')
plt.show()

#Obtention de la valeur prédite de E en combiant les deux modèles 
df['E'] = df['EVp']/df['volume_molaire']

r_sqr = metrics.r2_score(df['EV'], df['EVp'])

df['ecart_relatif_E'] = (df['E']-df["Young's Modulus at RT ( GPa )"])/df["Young's Modulus at RT ( GPa )"]*100
plt.scatter(df.E.values, df['ecart_relatif_E'].values)
plt.title("Ecart relatif module de Young")
plt.show()

plt.scatter(df['EV'], df['EVp'])
plt.title("Regression polynomiale de degré " + str(degree) + " du produit E* Vm,  R² =" + str(round(r_sqr, 3)))
plt.savefig('C:/Users/adelie.saule/MIG/Resultats_de_regressions/Regression_polynomiale_EVm.png')
plt.show()

print("coefficients regression E*Vm et intercept" )
print((modelEVm.steps[1][1].coef_))
print((modelEVm.steps[1][1].intercept_))



# Etant donné une composition, calcul du module de Young prédit

data = [[ 60+i, 40-i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(41)]

df1 = pd.DataFrame(data = data, index = None, columns = elements)
#SiO2 Na2O CaO Al2O3 B2O3 MgO PbO P2O5 Li2O K2O TiO2 BaO ZnO


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
    plt.xlabel(xlabel = 'Fracion molaire de Na2O')
    plt.ylabel(ylabel = "Module d'Young (GPa)")
    plt.savefig('C:/Users/adelie.saule/MIG/Resultats_de_regressions/prediction_E_verre_SiO2_Na2O.png')
    plt.show()
    
predictionE(df1)
    
def predictionrot(df):
    elements = df.columns
    df['volume_molaire_predit'] = modelVm.predict(df.to_numpy())
    df['M'] = sum(df[elem]*massmolaire(elem)/100 for elem in elements)
    df['rot'] = df['M']/df['volume_molaire_predit']
    # plt.plot(df.Na2O.values, df.rot.values)
    # plt.title("rho prédit pour un verre 100-x SiO2, x Na2O")
    # plt.grid()
    # plt.savefig('C:/Users/adelie.saule/MIG/Resultats_de_regressions/prediction_rho_verre_SiO2_Na2O.png')
    # plt.show()
    return df



dicordinence = {'Na2O' : 6, 'CaO' : 8, 'PbO' : 8, 'BaO' : 8, 'TiO2' : 6, 'GeO2' : 4, 'ZnO' : 6, 'MgO' : 6, 'Li2O' : 6, 'K2O' : 8, 'Al2O3' : 4, 'SiO2' : 4, 'B2O3' : 3 }
dicoliaison = {'SiO2' : 106, 'Na2O' : 20, 'CaO' : 32, 'B2O3' : 119, 'TiO2' : 73, 'PbO': 73, 'GeO2' : 108, 'ZnO' : 72, 'MgO' : 37, 'K2O' : 13, 'Al2O3' : 90, 'Li2O' : 36, 'BaO' : 33}
dicof = {'SiO2' : 1, 'Na2O' : 2, 'CaO' : 1, 'B2O3' : 2, 'TiO2' : 1, 'PbO' : 1, 'GeO2' : 1, 'ZnO' : 1, 'MgO':1, 'K2O' : 2, 'Al2O3' : 2, 'Li2O' : 2, 'BaO' : 1}
dicoO = {'SiO2' : 2, 'Na2O' : 1, 'CaO' : 1, 'B2O3' : 3, 'TiO2' : 2, 'PbO' : 1, 'GeO2' : 2, 'ZnO' : 1, 'MgO' : 1, 'K2O' : 1, 'Al2O3' : 3, 'Li2O' : 1, 'BaO' : 1}
dicor = {'SiO2' : 0.26, 'Na2O' : 1.02, 'CaO' : 1.12, 'B2O3' : 0.01, 'TiO2' : 0.605, 'PbO' : 1.29, 'GeO2' : 0.39, 'ZnO' : 0.74, 'MgO' : 0.72, 'K2O': 1.51, 'Al2O3' : 0.39, 'Li2O' : 0.76, 'BaO' : 1.42}

#↓les rayons sont en en °angström

N = 6.02*10**23

df = pd.read_csv('C:/Users/adelie.saule/MIG/Database/Y_50_130_D_2.2_3_V_3000_7500.csv', sep = ' ') 
print(len(df))

def modeleyammackenfort(df):
    elements = df.columns[:-3]
    
    df['Sum'] =  sum(df[elem] for elem in elements)
    df['M'] = sum(df[elem]*massmolaire(elem)/df['Sum'] for elem in elements)
    df['alpha1'] = sum(df[elem]*dicordinence[elem]*dicoliaison[elem]*dicof[elem] for elem in elements)
    df['alpha'] = df['alpha1']/sum(df[elem]*dicordinence[elem]*dicoliaison['SiO2']*dicof[elem] for elem in elements)
    df['V1'] = df['Density at RT ( g/cm3 )']/df['M']*4/3*m.pi*sum(dicor[elem]**3*N*dicof[elem]*df[elem]/df['Sum'] for elem in elements)*10**(-30)*10**6  
    df['Vo'] = df['Density at RT ( g/cm3 )']/df['M']*sum(df[elem]*dicoO[elem]*4/3*m.pi*N*1.4**3/df['Sum'] for elem in elements)*10**(-30)*10**6
    df['V'] = df['V1'] + df['Vo']
    df = df[(df['V'] < 0.55) ]
    print(len(df))
    df['Hv'] = 0.053*(df['alpha']/(0.520 + 0.052*df['V'] - df['V']**2))**(0.5)*df["Young's Modulus at RT ( GPa )"]*1000
    df['relative_error'] = (df['Hv'] - df["Vickers Hardness 100g ( MPa )"])/df["Vickers Hardness 100g ( MPa )"]*100
    plt.scatter(df["Vickers Hardness 100g ( MPa )"].values, df['relative_error'].values)
    plt.title("Erreur relative modèle dureté Vickers")
    plt.grid()
    plt.xlabel('dureté Vickers')
    plt.savefig('C:/Users/adelie.saule/MIG/Resultats_de_regressions/Erreur_relative_durete_Vickers.png')
    plt.show()
    
    plt.scatter(df["Vickers Hardness 100g ( MPa )"].values,df['Hv'].values)
    plt.title("Modèle dureté Vickers")
    plt.grid()
    plt.xlabel('dureté Vickers réelle')
    plt.ylabel('dureté Vickers modèle')
    plt.savefig('C:/Users/adelie.saule/MIG/Resultats_de_regressions/Modele_durete_Vickers.png')
    plt.show()



#║modeleyammackenfort(df)

data = [[ 70.41, 14.99, 7.4, 3.3, 0, 3.7, 0, 0, 0, 0, 0, 0, 0], [ 69.11, 14.99, 8.7, 3.3, 0, 3.7, 0, 0, 0, 0, 0, 0, 0],[ 69.11, 14.99, 7.4, 3.3, 0, 5.0, 0, 0, 0, 0, 0, 0, 0]]


df4 = pd.DataFrame(data = data, index = None, columns = elements)

def modeleyammackenfort_test(df):
    predictionrot(df)
    predictionE(df)
    elements = df.columns[:-3]
    print(df.head())
    print(elements)
    df['Sum'] =  sum(df[elem] for elem in elements)
    df['M'] = sum(df[elem]*massmolaire(elem)/df['Sum'] for elem in elements)
    df['alpha1'] = sum(df[elem]*dicordinence[elem]*dicoliaison[elem]*dicof[elem] for elem in elements)
    df['alpha'] = df['alpha1']/sum(df[elem]*dicordinence[elem]*dicoliaison['SiO2']*dicof[elem] for elem in elements)
    df['V1'] = df['rot']/df['M']*4/3*m.pi*sum(dicor[elem]**3*N*dicof[elem]*df[elem]/df['Sum'] for elem in elements)*10**(-30)*10**6  
    df['Vo'] = df['rot']/df['M']*sum(df[elem]*dicoO[elem]*4/3*m.pi*N*1.4**3/df['Sum'] for elem in elements)*10**(-30)*10**6
    df['V'] = df['V1'] + df['Vo']
    print(df.head(3))
    df = df[(df['V'] < 0.55) ]
    print(len(df))
    df['Hv'] = 0.053*(df['alpha']/(0.520 + 0.052*df['V'] - df['V']**2))**(0.5)*df["Young's Modulus at RT ( GPa )"]*1000
    df['relative_error'] = (df['Hv'] - df["Vickers Hardness 100g ( MPa )"])/df["Vickers Hardness 100g ( MPa )"]*100
    print(df.head(2))
    return df
    # plt.scatter(df["Vickers Hardness 100g ( MPa )"].values, df['relative_error'].values)
    # plt.title("Erreur relative modèle dureté Vickers")
    # plt.grid()
    # plt.xlabel('dureté Vickers')
    # plt.savefig('C:/Users/adelie.saule/MIG/Resultats_de_regressions/Erreur_relative_durete_Vickers.png')
    # plt.show()
    
    # plt.scatter(df["Vickers Hardness 100g ( MPa )"].values,df['Hv'].values)
    # plt.title("Modèle dureté Vickers")
    # plt.grid()
    # plt.xlabel('dureté Vickers réelle')
    # plt.ylabel('dureté Vickers modèle')
    # plt.savefig('C:/Users/adelie.saule/MIG/Resultats_de_regressions/Modele_durete_Vickers.png')
    # plt.show()



# modeleyammackenfort_test(df4)



# # modèle de dureté Vickers avec le modèle pour E et non le E expérimental
# df2 = pd.read_csv('C:/Users/adelie.saule/MIG/Database/compoV_v2.csv', sep = ' ')
# df3 = pd.read_csv('C:/Users/adelie.saule/MIG/Database/V_v2.csv', sep = ' ')

# def modeleyammackenfort_a_partir_composition(df):
    
#     df = prediction_E_et_rho(df)
    
#     print(df.head(20))

#     elements = df.columns[:-5]
    
#     print(elements)
    
#     df['Sum'] =  sum(df[elem] for elem in elements)
#     df['M'] = sum(df[elem]*massmolaire(elem)/df['Sum'] for elem in elements)
#     df['alpha1'] = sum(df[elem]*dicordinence[elem]*dicoliaison[elem]*dicof[elem] for elem in elements)
#     df['alpha'] = df['alpha1']/sum(df[elem]*dicordinence[elem]*dicoliaison['SiO2']*dicof[elem] for elem in elements)
#     df['V1'] = df['rho']/df['M']*4/3*m.pi*sum(dicor[elem]**3*N*dicof[elem]*df[elem]/df['Sum'] for elem in elements)*10**(-30)*10**6  
#     df['Vo'] = df['rho']/df['M']*sum(df[elem]*dicoO[elem]*4/3*m.pi*N*1.4**3/df['Sum'] for elem in elements)*10**(-30)*10**6
#     df['V'] = df['V1'] + df['Vo']
#     df = df[(df['V'] < 0.55) ]
#     print(len(df))
#     df['Hv'] = 0.053*(df['alpha']/(0.520 + 0.052*df['V'] - df['V']**2))**(0.5)*df["Young's Modulus at RT ( GPa )"]*1000
#     df['relative_error'] = (df['Hv'] - df3["Vickers Hardness 100g ( MPa )"])/df3["Vickers Hardness 100g ( MPa )"]*100
#     print(df["relative_error"])
#     plt.scatter(df3["Vickers Hardness 100g ( MPa )"].values, df['relative_error'].values)
#     plt.title("Erreur relative modèle dureté Vickers")
#     plt.grid()
#     plt.xlabel('dureté Vickers')
#     plt.show()


    
# modeleyammackenfort_a_partir_composition(df2)
