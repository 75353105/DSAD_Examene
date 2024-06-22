import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score,silhouette_samples

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print("A")
print("------> Cerinta 1------->")
df_vot = pd.read_csv("Vot.csv", index_col=None)
df_coduri = pd.read_csv("CoduriLocalitat.csv", index_col=None)

lista_procente = list(df_vot)[2:]
lista_procente_barbati = list(df_vot)[2:8]
print(lista_procente_barbati)
lista_procente_femei = list(df_vot)[8:]
print(lista_procente_femei)
cerinta1 = df_vot.copy()
cerinta1['Barbati'] = 0
cerinta1['Femei'] = 0
for procent in lista_procente_barbati:
    cerinta1['Barbati'] += cerinta1[procent]
for procent in lista_procente_femei:
    cerinta1['Femei'] += cerinta1[procent]
cerinta1.to_csv("intermediar.csv")
cerinta1_final = cerinta1[cerinta1['Femei'] > cerinta1['Barbati']]
cerinta1_final.to_csv("Cerinta1.csv")

print("------> Cerinta 2------>\n")
df_intermediar = df_vot[['Siruta'] + lista_procente]
merger = df_intermediar.merge(df_coduri, on='Siruta')
merged = merger.groupby('Judet')[lista_procente].mean()
merged.to_csv("Cerinta2.csv")

print("B\n")
print("Cerinta 1")
#Componenta partitiei din nrc clusteri
nrc = input("Introduceti numarul de clusteri dorit: ")
nrc = int(nrc)
df_cluster = df_vot[lista_procente]

clustering = AgglomerativeClustering(n_clusters=nrc).fit(df_cluster)
df_partitie = pd.DataFrame(clustering.labels_, index=df_cluster.index, columns=['Cluster'])
df_partitie.to_csv("p.csv")

print("Cerinta 2")
#Histograma pentru clusterii din partitia cu nrc clusteri
classes = np.unique(df_partitie['Cluster'])
for col in df_cluster.columns:
    min_max = (df_cluster[col].min(), df_cluster[col].max())
    fix, ax = plt.subplots(1, len(classes), figsize=(12, 10), sharey=True)
    fix.suptitle("Histograma " + col)
    for i, cls in enumerate(classes):
        ax[i].hist(df_cluster[df_partitie['Cluster'] == cls][col], range=min_max, rwidth=0.9)
        ax[i].set_xlabel(str(cls))
plt.show()