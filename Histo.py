import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/raphael.poux/Mig verre/Interglad_post_traitement.csv")

def garde_Young(df, inf, sup):
    """
    Méthode qui supprime les lignes où le module d'Young à temp. ambiante n'est pas renseigné
    On peut choisir 50 et 130 pour les bornes
    """
    df['Young_check at RT ( GPa )'] = (df["Young's modulus at RT ( GPa )"] > inf) & (df["Young's modulus at RT ( GPa )"] <= sup)
    df = df.loc[df['Young_check at RT ( GPa )'] == True]

def garde_Vickers(df, inf, sup):
    """
    Méthode qui supprime les lignes où le module d'Young à temp. ambiante n'est pas renseigné
    On peut choisir 3000 et 7500 pour les bornes
    """
    df['Vickers Hardness 100g_check ( MPa )'] = (df["Vickers Hardness 100g ( MPa )"] > inf) & (df["Vickers Hardness 100g ( MPa )"] <= sup)
    df = df.loc[df['Vickers Hardness 100g_check ( MPa )'] == True]

def garde_densite(df, inf, sup):
    """
    Méthode qui supprime les lignes où le module d'Young à temp. ambiante n'est pas renseigné
    On peut choisir 2 et 4 pour les bornes
    """
    df['Density at RT_check ( g/cm3 )'] = (df["Density at RT ( g/cm3 )"] > inf) & (df["Density at RT ( g/cm3 )"] <= sup)
    df = df.loc[df['Density at RT_check ( g/cm3 )  '] == True]

def garde_toughness(df, inf, sup):
    """
    Méthode qui supprime les lignes où le module d'Young à temp. ambiante n'est pas renseigné
    On peut choisir 0.5 et 1.5 pour les bornes
    """
    df['Fracture Toughness_check ( MPa.m1/2 )'] = (df["Fracture Toughness ( MPa.m1/2 )"] > inf) & (df["Fracture Toughness ( MPa.m1/2 )"] <= sup)
    df = df.loc[df['Fracture Toughness_check ( MPa.m1/2 )'] == True]

def Vickers(df, inf, sup, bin):
  df = garde_Vickers(df, inf, sup).loc[df["Vickers Hardness (Typical) ( MPa )"] != 0]
  print(len(df))
  hist = df.hist(column='Vickers Hardness 100g_check ( MPa )', bins = bin)
  plt.show()

def Young(df, inf, sup, bin):
  df = garde_Young(df, inf, sup).loc[df["Young's Modulus at RT ( GPa )"] != 0]
  print(len(df))
  hist = df.hist(column='Young_check at RT ( GPa )', bins = bin)
  plt.show()

def Toughness(df, inf, sup, bin):
  df = df.loc[df["Fracture Toughness ( MPa.m1/2 )"] != 0]
  print(len(df))
  hist = df.hist(column='Fracture Toughness_check ( MPa.m1/2 )', bins = 30)
  plt.show()

def Density(df, inf, sup, bin):
  df = garde_densite(df, inf, sup).loc[df["Density at RT ( g/cm3 )"] != 0]
  print(len(df))
  hist = df.hist(column='Density at RT_check ( g/cm3 )', bins = 30)
  plt.show()
