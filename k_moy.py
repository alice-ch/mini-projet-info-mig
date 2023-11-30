# Algorithme des K moyennes
import pandas as pd

df = pd.read_csv("C:/Users/raphael.poux/Mig verre/CompoD.csv", sep = ' ')
dfD = pd.read_csv("C:/Users/raphael.poux/Mig verre/D.csv", sep = ' ')
#test = pd.read 


def k_moyennes(k , df):
    n = len(df)
    elements = df.columns
    dico={}
    h = len(df) // k
    df['groupe'] = pd.Series(0, range(n))
    dico2 = {}
    final = df.columns
    for i in range(k):
        dico[i] = tuple([df[elem].iloc[h*i] for elem in elements])
        df[f"{i}"] = pd.Series(0, range(n))
    while dico2 != dico:
        dico2 = dico.copy()
        df, dico, k, elements, n = truc_iterable(df, dico, k, elements, n)
    #df[final].to_csv('kgroupes.csv', sep = ',',index=False)
    dfD[df['groupe'] == 1].hist()
        
def truc_iterable(df, dico, k, elements, n):
    for i in range(k):
        df[f'{i}'] = (((df[elements] - dico[i])**2)**(1/2)).sum(axis = 1)
    df['min'] = df[[f'{i}' for i in range(k)]].min(axis = 1)
    df['groupe'] = sum(i*(df[f'{i}'] == df['min']) for i in range(k))
    for i in range(k):
        D = df[df['groupe'] == i].mean()
        dico[i] = tuple([D[elem] for elem in elements])
    return (df, dico, k, elements, n)


    
k_moyennes(60 , df)
