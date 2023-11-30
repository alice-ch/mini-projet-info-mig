
import pandas as pd

df1 = pd.read_csv("C:/Users/raphael.poux/Mig Verre/100000.csv",header = 28)
df2 = pd.read_csv("C:/Users/raphael.poux/Mig Verre/200000.csv",header = 28)
df3 = pd.read_csv("C:/Users/raphael.poux/Mig Verre/350000.csv", header = 28)
df4 = pd.read_csv("C:/Users/raphael.poux/Mig Verre/500000.csv", header = 28)

df1.columns
df4.columns
df3.columns
df2.columns

dico = {}
for i in df1.columns:
    dico[i] = True
for i in df2.columns:
    dico[i] = True
for i in df3.columns:
    dico[i] = True
for i in df4.columns:
    dico[i] = True

colonnes = [key for key in dico]

for colonne in colonnes:
    if not colonne in df1.columns:
        df1[colonne] = pd.Series(0)
        
for colonne in colonnes:
    if not colonne in df2.columns:
        df2[colonne] = pd.Series(0)
        
for colonne in colonnes:
    if not colonne in df3.columns:
        df3[colonne] = pd.Series(0)
        
for colonne in colonnes:
    if not colonne in df4.columns:
        df4[colonne] = pd.Series(0)
        
df1.to_csv('Interglad.csv', sep = ',',index = False)
df2.to_csv('Interglad.csv', sep = ',', mode = 'a',index = False)
df3.to_csv('Interglad.csv', sep = ',', mode = 'a',index = False)
df4.to_csv('Interglad.csv', sep = ',', mode = 'a',index = False)

df = pd.read_csv("Interglad.csv")
colomns = {}

for i in df.columns:
    if i[0] == " ":
        colomns[i] = i[1:]
    else:
        colomns[i] = i  

df = df.rename(columns = colomns)

'''
for i in range(5,60000):
    if df['SiO2'].iloc[i] == ' SiO2':
 '''
       
df.drop([25343,47698,80098], inplace=True)
        


df.to_csv('Interglad_propre_test.csv', sep = ',',index=False)
