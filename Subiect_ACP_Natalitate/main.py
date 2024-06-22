import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

print("A\n")
print("----->Cerinta 1------>")
df_natalitate = pd.read_csv("MiseNatPopTari.csv", index_col=None)
df_coduri = pd.read_csv("CoduriTariExtins.csv", index_col=None)

df_sporuri_negative = df_natalitate[df_natalitate['RS']<0]
df_sporuri_negative.to_csv("Cerinta1.csv")

print("------>Cerinta 2------>\n")
valori_numerice = list(df_natalitate)[3:]
df_intermediar = df_natalitate.merge(df_coduri, on="Country_Number")
cerinta2 = df_intermediar[valori_numerice + ['Continent']].groupby("Continent").mean()
cerinta2.to_csv("Cerinta2.csv")

print("B")
print("------>Cerinta 1------>")
#Varianta componentelor principale, se vor lua doar valorile numerice
df_ACP = df_natalitate[valori_numerice].copy()
acp = PCA()
acp.fit(df_ACP)

varianta = acp.explained_variance_
procent_variante = acp.explained_variance_ratio_ * 100
varianta_cumulata = acp.explained_variance_.cumsum()
procent_cumulat = acp.explained_variance_ratio_.cumsum() * 100

df_varianta = pd.DataFrame(
    {
        "Varianta": varianta,
        "Procenta varianta": procent_variante,
        "Varianta cumulata": varianta_cumulata,
        "Procent cumulat": procent_cumulat
    }
)
print(df_varianta)

print("------>Cerinta 2------>")
#Scorurile asociate instantelor (Componentelor principale)
df_scoruri = pd.DataFrame(acp.transform(df_ACP), df_ACP.index, df_varianta.index)
df_scoruri.to_csv("scoruri.csv")

print("------>Cerinta 3------>")
#Graficul scorurilor pe primele doua axe principale (asta inseamna ca se pun in grafic numai primele doua componente principale din cele sapte)
plt.figure(figsize=(12, 10))
plt.title("Scorurile primelor doua componente principale")
plt.xlabel("CP1")
plt.ylabel("CP2")
plt.axvline(0)
plt.axhline(0)
plt.scatter(df_scoruri[0], df_scoruri[1])
plt.show()
