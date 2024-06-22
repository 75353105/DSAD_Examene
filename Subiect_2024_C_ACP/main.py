import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print("A\n")
print("------>Cerinta 1------>")
df_indicatori = pd.read_csv("GlobalIndicatorsPerCapita_2021_Extended.csv", index_col=None)
df_coduri = pd.read_csv("CountryCodes_Extended.csv", index_col=None)
lista_valori = list(df_indicatori)[9:]

cerinta1 = df_indicatori[['Cod','Tara']].copy()
cerinta1['Valoare_Adaugata'] = 0

for valoare in lista_valori:
    cerinta1['Valoare_Adaugata'] += df_indicatori[valoare]

cerinta1.to_csv("Cerinta1.csv")
print("------>Cerinta 2------>\n")
#Bun, e buna practica pana scrii mai departe sa vezi cum arata setul grupat si agregat
#In cazul de fata, Continent a devenit un index_col fara sa specific si nu poate fi identificat
#El va fi introdus automat daca salvez fisierul insa, asa ca o sa pun in cerinta2 doar coloanele de coeficienti
lista_coeficienti = list(df_indicatori)[2:]
df_intermediar = df_indicatori.merge(df_coduri, on='Cod')
t = df_intermediar.groupby('Continent').sum()
cerinta2 = t[lista_coeficienti].copy()
cerinta2.to_csv("Cerinta2.csv")

print("B\n")
print("------>Cerinta 1------>")
#Se cere variatia componentelor principale, se vor luat doar coloanele numerice din setul de date df_indicatori
df_ACP = df_indicatori[lista_coeficienti].copy()
acp = PCA()
acp.fit(df_ACP)

varianta = acp.explained_variance_
procent_varianta = acp.explained_variance_ratio_ * 100
varianta_cumulata = acp.explained_variance_.cumsum()
procent_cumulat = acp.explained_variance_ratio_.cumsum() * 100
df_varianta = pd.DataFrame({
    "Varianta componenta": varianta,
    "Procenta varianta componenta": procent_varianta,
    "Varianta cumulata": varianta_cumulata,
    "Procente cumulate": procent_cumulat
})
print(df_varianta)

print("------>Cerinta 2------->")
#Scorurile componentelor principale
df_scoruri = pd.DataFrame(acp.transform(df_ACP), df_ACP.index, df_varianta.index)
df_scoruri.to_csv("scoruri.csv")

print("------>Cerinta 3------>")
plt.figure(figsize=(12, 10))
plt.title("Scorurile in primele doua axe") #Se adauga numai primele doua componente prinncipale
plt.xlabel("CP1")
plt.ylabel("CP2")
plt.axhline(0)
plt.axvline(0)
plt.scatter(df_scoruri[0], df_scoruri[1])
plt.show()