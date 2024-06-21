import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

#Subiect pana la nota 7, ultima cerinta este grea
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("A\n")
print("------>Cerinta 1------>\n")
df_alcool = pd.read_csv('alchol.csv', index_col=None)
df_tari = pd.read_csv("CoduriTariExtins.csv", index_col=None)
lista_ani = list(df_alcool)[2:]
print(lista_ani)
cerinta1 = df_alcool.copy()
cerinta1.insert(7, 'Suma', 0)
cerinta1.insert(8, 'Media', 0)
ani = ['2000', '2005', '2010', '2015', '2018']

for an in ani:
    cerinta1['Suma'] += cerinta1[an]

cerinta1['Media'] = cerinta1['Suma']/5.0

raspuns = cerinta1[['Code','Media']].copy()
raspuns.to_csv("Cerinta1.csv")

print("------>Cerinta 2------>")
df_jonctiune = df_alcool.merge(df_tari, on='Code')
t2 = df_jonctiune.groupby('Continent')[ani].mean()
t2['Anul'] = False

for index, row in t2.iterrows():
    max_value = row[ani].max()
    max_year = row[ani].idxmax()
    t2.at[index, 'Anul'] = max_year

#Daca faci print t2 o sa vezi ca termenul Continent e sub ani, asta inseamna ca el este un index de coloana si se include automat daca faci o copie de tabel
cerinta2 = t2[['Anul']].copy()
cerinta2.to_csv("Cerinta2.csv")

print("B\n")
print("------>Cerinta 1------>")
#Matricea ierarhie
df_consum_ani = df_alcool[lista_ani].copy()
Z = linkage(df_consum_ani, 'ward')
matrice_ierarhica = pd.DataFrame(Z, columns=['ID1', 'ID2', 'Distanta', 'Nr de elemente in cluster'])
print(matrice_ierarhica)

print("------>Cerinta 2------>")
#Dendrograma partitia optimala
#Numar optim de clustere
m = Z.shape[0]
d = Z[1:, 2] - Z[:len(Z) - 1, 2]
k = m - np.argmax(d)
#k este numarul optim de clusteri

#Trebuie creata partitia de clustere
clustering = AgglomerativeClustering(n_clusters=int(k)).fit(df_consum_ani)
df_partitie = pd.DataFrame(clustering.labels_, index=df_consum_ani.index, columns=['Cluster'])

#Se calculeaza scorurile Silhouette pentru a determina partitia optimala
scor_Silhouette = silhouette_score(df_consum_ani, df_partitie['Cluster'])
print(scor_Silhouette)

#Dendrograma
plt.figure(figsize=(15, 10))
plt.title("Dendograma")
plt.xlabel("An")
plt.ylabel("Alcool")
dendrogram(Z, truncate_mode='lastp', p=int(k), leaf_rotation=75., labels=df_consum_ani.index)
plt.show()
