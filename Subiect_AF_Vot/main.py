import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity,calculate_kmo

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print("A")
print("------>Cerinta 1------>")
df_vot = pd.read_csv("Vot.csv", index_col=0)
df_judete = pd.read_csv("Coduri_localitati.csv", index_col=0)

#valori_procentuale se foloeste la prima parte, valori_procentuale_2 la analiza factoriala, e nevoie de el ca sa nu am gauri nemodificabile in tabel
#Fisierul Vot.csv a fost realizat prost
valori_procentuale = list(df_vot)[1:]
valori_procentuale_2 = list(df_vot)[1:-1]
print(valori_procentuale)
t1 = df_vot.copy()
t1['Categorie'] = True

#Parcurgere astfel incat sa aflii valorile minime, index pt coloana si row pentru inregistrari
for index, row in t1.iterrows():
    prezenta_minima = row[valori_procentuale].min()
    categorie = row[valori_procentuale].idxmin()
    t1.at[index, 'Categorie'] = categorie
cerinta1 = t1[['Localitate', 'Categorie']].copy()
cerinta1.to_csv("Cerinta1.csv")

print("------>Cerinta 2------>\n")
df_jonctiune = df_vot.merge(df_judete, on='Localitate')
#Daca ai medie la jonctiune faci cu mean() nu cu sum()
t2 = df_jonctiune.groupby('Judet')[valori_procentuale].mean()
cerinta2 = t2[valori_procentuale].copy()
cerinta2.to_csv("Cerinta2.csv")

print("B\n")
print("------>Cerinta 1------>")
#Testul Bartlett de relevanta
#Vezi ca tabelul Vot.csv e prost scris, trebuie sa fac o schema ca sa il fac sa functioneze ca e plin de valori NaN
df_analiza_factoriala = df_vot[valori_procentuale_2]
df_analiza_factoriala.fillna(df_analiza_factoriala.mean(), inplace=True)
chi_square_value, p_value = calculate_bartlett_sphericity(df_analiza_factoriala)
if(p_value > 0.001):
    print("Nu se poate aplica analiza factoriala: p_value=", p_value)
else:
    print("Se va putea aplica analiza factoriala: p_value=", p_value)

print("------> CERINTA 2------>")
#Calcul scoruri factoriale
#Se creeaza modelul de analiza factoriala
#Trebuie sa faci si pasul de fa.fit desi Nicu nu l-a pus ca bulangiul ce este
fa = FactorAnalyzer(n_factors=df_analiza_factoriala.shape[1], rotation=None)
fa.fit(df_analiza_factoriala)
#Se calculeaza scorurile factoriale
df_scoruri_factoriale = pd.DataFrame(fa.transform(df_analiza_factoriala), index=df_analiza_factoriala.index,
                                     columns=["F" + str(i) for i in range(1, df_analiza_factoriala.shape[1] + 1)])
df_scoruri_factoriale.to_csv("f.csv")

print("------> Cerinta 3------>")
#Graficul scorurilor factoriale pentru primii doi factori
plt.figure(figsize=(20, 15))
plt.title("Grafic primele doua scoruri")
plt.xlabel("F1")
plt.ylabel("F2")
plt.axvline(0)
plt.axhline(0)
plt.scatter(df_scoruri_factoriale["F1"], df_scoruri_factoriale["F2"])
#Asta pui ca sa ai codul Siruta in grafic, nu stiu daca este necesar
for i in range(len(df_scoruri_factoriale)):
    plt.text(df_scoruri_factoriale["F1"].iloc[i], df_scoruri_factoriale["F2"].iloc[i], df_scoruri_factoriale.index[i])

plt.show()