import numpy as np
import pandas as pd
from math import *

df = pd.read_csv('D:/Users/Valentin/Desktop/1A Mines/Semestre 1/UE15 - MIG/mini-projet-info-mig/Interglad.csv')

# On garde que les verres

# On vire les colonnes pas utiles
liste_col_a_enlever = ['Unnamed: 0', 'No.', ' Glass No.', ' Data Source', ' Year', ' Data Source Number', ' Thermal Treatment',
                       ' Bulk Density ( g/cm3 )', ' Density (Miscell) ( g/cm3 )',  " Young's Modulus (Miscell) ( GPa )", " Young's Modulus at <0C ( GPa )",
                       ' Vickers Hardness 50g ( MPa )', ' Vickers Hardness 100g ( MPa )', ' Vickers Hardness 200g ( MPa )', ' Vickers Hardness 500g ( MPa )', ' Vickers Hardness (Miscell) ( MPa )',
                       ' H', ' Li', ' B', ' C', ' N', ' O', ' F', ' Na', ' Mg', ' Al', ' Si', ' P', ' S', ' Cl', ' K', ' Ca', ' Sc', ' Ti', ' V', ' Cr', ' Mn', ' Fe', ' Co', ' Ni', ' Cu', ' Zn', ' Ga', ' Ge', ' As', ' Se', ' Br', ' Rb', ' Y', ' Zr', ' Nb', ' Mo', ' Pd', ' Ag', ' Cd', ' In', ' Sn', ' Sb', ' Te', ' I', ' Ba', ' La', ' Ce', ' Gd', ' W', ' Re', ' Pt', ' Au', ' Hg', ' Tl', ' Pb', ' Bi', ' Th', ' U',
                       ]

df.drop(liste_col_a_enlever, axis = 1, inplace=True)        # Enlève les colonnes inutiles


# On convertit tout en float
df = df.replace(' *', np.nan)               # * -> NaN
df = df.replace(' ', np.nan)                # ' ' -> NaN
df = df.fillna(0)                           # NaN -> 0

for i in df.columns:
    df[i] = pd.to_numeric(df[i], errors='coerce')                       # type object -> type float

df = df.fillna(0)                           # NaN -> 0

df = df[df['SiO2'] > 0]                     # on ne garde que les verres silicatés


print(df.head(5))

Liste_tous_composants = [ ' SiO2', ' B2O3', ' Al2O3', ' MgO', ' CaO', ' BaO', ' Li2O', ' Na2O', ' K2O', ' Cu2O', ' Rb2O', ' Ag2O', ' Cs2O', ' Tl2O', ' BeO', ' MnO', ' FeO', ' CoO', ' NiO', ' CuO', ' ZnO', ' SrO', ' CdO', ' PbO', ' HgO', ' SnO', ' Cr2O3', ' Fe2O3', ' Ga2O3', ' As2O3', ' Y2O3', ' In2O3', ' Sb2O3', ' La2O3', ' Nd2O3', ' Gd2O3', ' Bi2O3', ' Co2O3', ' Sc2O3', ' Co3O4',
                          ' Sb2O5', ' Ce2O3', ' As2O5', ' TiO2', ' MnO2', ' CoO2', ' GeO2', ' ZrO2', ' CeO2', ' SnO2', ' TeO2', ' P2O5', ' V2O5', ' Nb2O5', ' Ta2O5', ' V2O4', ' Mo2O5', ' HfO2', ' SO3', ' MoO3', ' WO3', ' SO2', ' GeS2', ' H2O', ' OH', ' NH3', ' N2O5', ' Sm2O3', ' Eu2O3', ' Tb2O3', ' Dy2O3', ' Ho2O3', ' Er2O3', ' Tm2O3', ' Yb2O3', ' Lu2O3', ' Pr2O3', ' Pr6O11', ' PrO2',
                          ' ThO2', ' U2O3', ' UO2', ' U3O8', ' Ti2O3', ' R2O', ' RO', ' R2O3', ' RO2', ' RO3', ' RE2O3', ' LiF', ' NaF', ' KF', ' RbF', ' CsF', ' TlF', ' BeF2', ' MgF2', ' CaF2', ' SrF2', ' BaF2', ' MnF2', ' CuF2', ' ZnF2', ' CdF2', ' SnF2', ' PbF2', ' ScF3', ' YF3',
                          ' CrF3', ' AlF3', ' GaF3', ' InF3', ' LaF3', ' NdF3', ' GdF3', ' YbF3', ' LuF3', ' TiF4', ' ZrF4', ' HfF4', ' ThF4', ' VF3', ' KHF2', ' DyF3', ' HoF3', ' CoF2', ' FeF3', ' VF4', ' NiF2', ' UF4', ' TbF3', ' EuF3', ' LiCl', ' NaCl', ' MgCl2', ' CaCl2', ' SrCl2', ' BaCl2', ' CuCl2', ' ZnCl2', ' PbCl2', ' AgCl', ' VCl3', ' LiBr', ' NaBr', ' CsBr', ' CdBr2', ' PbBr2', ' AgBr', ' NaI', ' CsI', ' SrI2', ' PbI2', ' AgI', ' PF5', ' TmF3', ' ErF3', ' PrF3', ' P2S5',
                          ' Mn3O4', ' Fe3O4', ' SeO2', ' RuO2', ' CdS', ' Li2S', ' ZnS', ' ZnSe', ' Ni2O3', ' CdSe', ' CdTe', ' R2O5', ' BiF3', ' CeF3', ' SiF4', ' SmF3', ' KCl', ' RbCl', ' CsCl', ' CdCl2', ' SnCl2', ' BiCl3', ' ThCl4', ' CuCl', ' PrCl3', ' KBr', ' RbBr', ' MgBr2', ' SrBr2', ' BaBr2', ' CuI', ' NbF5', ' Na2S', ' Rb2S', ' Sr', ' Cs', ' Pr', ' Nd', ' Sm', ' Eu', ' Tb', ' Dy', ' Ho', ' Er', ' Tm', ' Yb', ' Lu', ' D', ' Ca(PO3)2', ' CaSO4', ' K2CO3', ' K2CrO7', ' Mg(PO3)2', ' Na4P2O7', ' Zn(PO3)2', ' AlPO4', ' AgNO3', ' Li2O-Al2O3-4SiO2', ' Na2B4O7-10H2O', ' NaHPO4-12H2O', ' Li2SO4', ' La(PO3)3', ' BaTiO3', ' BPO4', ' Sr3(PO4)2', ' CaHPO4-2H2O', ' Na2SiF6', ' Ag2Se', ' Ga2S3', ' La2S3', ' GeSe3', ' K2S', ' ZrS2', ' Sb2Se3', ' ZrO2-SiO2', ' 2MgO-2Al2O3-5SiO2', ' 2Al2O3-SiO2', ' TiO', ' CaO-SiO2', ' Na2B4O7', ' GaAs', ' Pb3O4', ' V2O3', ' Nb2O3', ' UO3', ' Tb4O7', ' MnCl2', ' AlCl3', ' ErCl3', ' TeCl4', ' WCl6', ' TlBr', ' CaBr2', ' ZnBr2', ' LiI', ' KI', ' RbI', ' TlI', ' ZnI2', ' CdI2', ' BiI3', ' SiS2', ' PbO2', ' Be', ' Ru', ' Rh', ' Hf', ' Ta', ' PbGeO3', ' Li3PO4', ' Ag2SO4', ' In(PO3)3', ' K2B4O7', ' Sr(PO3)2', ' BaCO3', ' Li2CO3', ' H3BO3', ' Zn3(PO4)2', ' AgPO3', ' Li2B4O7', ' PbSO4', ' BaS', ' Nd2S3', ' Sb2S3', ' Ga2Se3', ' B2S3', ' Cs2S', ' Tm2S3', ' GeS', ' Er2S3', ' In2S3', ' PdO', ' Mn2O3', ' CrO3', ' EuO', ' MoO2', ' Rh2O3', ' NdCl3', ' CeBr3', ' GdBr3', ' HgI2', ' CoCl2', ' Na2Se', ' Ir', ' Lr', ' Li3BO3', ' Kaolin', ' Al(OH)3', ' BaSO4', ' H3PO4', ' MgCO3', ' SrCO3', ' K2O-Al2O3-6SiO2', ' Bi2S3', ' Tl2S', ' Cu2Se', ' SnS']


# tri de la liste pour avoir une liste de couples (oxydes, pourcentages de présence) avec un pourcentage de présence décroissant

def fusion(liste1,liste2):
    liste=[]
    i,j=0,0
    while i<len(liste1)and j<len(liste2):
        if liste1[i][1]>=liste2[j][1]:
            liste.append(liste1[i])
            i+=1
        else:
            liste.append(liste2[j])
            j+=1
    while i<len(liste1):
        liste.append(liste1[i])
        i+=1
    while j<len(liste2):
        liste.append(liste2[j])
        j+=1
    return liste


def tri_fusion(liste):
    if len(liste)<2:
        return liste[:]
    else:
        milieu=len(liste)//2
        liste1=tri_fusion(liste[:milieu])
        liste2=tri_fusion(liste[milieu:])
        return fusion(liste1,liste2)

# On drop les duplicatas

df.drop_duplicates(keep='first', inplace=True)                         # Les duplicatas parfaits
df.drop_duplicates(subset=Liste_tous_composants, keep=False, inplace=True)       # Les compos identiques

# selection d'oxydes
Liste_oxyde_proportion = []
n = len(df)
for x in Liste_tous_composants:
    Liste_oxyde_proportion.append( (x, df[x][df[x] > 0.001].count()/n) )
print(tri_fusion(Liste_oxyde_proportion))

def coupe_liste_triee(L, proportion_apparition_minimum):
    Oxydes_gardes = []
    Oxydes_sortis = []
    for i in range(len(L)):
        if L[i][1] < proportion_apparition_minimum:
            Oxydes_gardes = L[:i]
            Oxydes_sortis = L[i:]
            return Oxydes_gardes, Oxydes_sortis
    return L, []

Oxydes_sortis = coupe_liste_triee(tri_fusion(Liste_oxyde_proportion), 0.1)[1]

for (x,i) in Oxydes_sortis:
    df.drop(x, axis = 1, inplace=True)

#liste d'oxydes de Raviner 2020 pour comparaison
# SiO2, B2O3, Al2O3, MgO, CaO, BaO, Li2O,
# Na2O, K2O, Ag2O, Cs2O, Tl2O, BeO, NiO, CuO, ZnO, CdO, PbO,
# Ga2O3, Y2O3, La2O3, Gd2O3, Bi2O3, TiO2, ZrO2, TeO2, P2O5, V2O5,
# Nb2O5, Ta2O5, MoO3, WO3, H2O, Sm2O3, MgF2, PbF2, PbCl2

df['sum_check'] = (df['Sum'] > 98) & (df['Sum'] <= 100)

df = df.loc[df['sum_check'] == True]

print(df.head(5))
