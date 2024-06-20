import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import normalize

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print("------>CERINTA 1------->")
df_emisii = pd.read_csv("Emissions.csv", index_col=None)
cerinta1=pd.DataFrame()
cerinta1["CountryCode"] = df_emisii["CountryCode"]
cerinta1["Country"] = df_emisii["Country"]
cerinta1["Emisii_total_tone"] = (df_emisii["AirEmiss"]+df_emisii["Sulphur"]+df_emisii["Nitrogen"]+df_emisii["Ammonia"]+
                                 df_emisii['NonMeth']+df_emisii["Partic"]+df_emisii["GreenGE"]*1000+df_emisii["GreenGIE"]*1000)
cerinta1.to_csv("Cerinta1.csv")

print("------>CERINTA 2------>")
df_regiuni = pd.read_csv("populatieEuropa.csv", index_col=None)
cerinta2 = pd.DataFrame()
intermediar_jonctiune = df_emisii.merge(right = df_regiuni, left_index=True, right_index=True)
cerinta2 = intermediar_jonctiune.groupby("Region").sum()
cerinta2.drop(['CountryCode','Country_x'], axis=1, inplace=True)
cerinta2["AirEmiss"] = cerinta2["AirEmiss"]/cerinta2["Population"]*100000
cerinta2["Sulphur"] = cerinta2["Sulphur"]/cerinta2["Population"]*100000
cerinta2["Nitrogen"] = cerinta2["Nitrogen"]/cerinta2["Population"]*100000
cerinta2["Ammonia"] = cerinta2["Ammonia"]/cerinta2["Population"]*100000
cerinta2["NonMeth"] = cerinta2["NonMeth"]/cerinta2["Population"]*100000
cerinta2["Partic"] = cerinta2["Partic"]/cerinta2["Population"]*100000
cerinta2["GreenGE"] = cerinta2["GreenGE"]/cerinta2["Population"]*100000
cerinta2["GreenGIE"] = cerinta2["GreenGIE"]/cerinta2["Population"]*100000
cerinta2.to_csv("Cerinta2.csv")

print("B")
print("------>CERINTA 1------>")
df_productie = pd.read_csv("EnergyProduction.csv", index_col=0)
df_energii= df_productie[['Coal', 'Oil', 'Gas', 'BioOil']]
df_particule = df_emisii[['AirEmiss', 'Sulphur', 'Nitrogen', 'Ammonia', 'NonMeth', 'Partic', 'GreenGE', 'GreenGIE']]

n, p, q = len(df_energii), df_energii.shape[1], df_particule.shape[1]
m = min(p, q)
cca = CCA(n_components=m).fit(df_energii, df_particule)

z, u = cca.transform(df_energii, df_particule)
z, u = normalize(z, axis=0), normalize(u, axis=0)
etichete_z = ["z" + str(i+1) for i in range(m)]
etichete_u = ["u" + str(i+1) for i in range(m)]

df_z = pd.DataFrame(z, df_energii.index, etichete_z)
df_u = pd.DataFrame(u, df_particule.index, etichete_u)
df_z.to_csv("z.csv")
df_u.to_csv("u.csv")

print("------>CERINTA 2------>")
r = np.diag(np.corrcoef(z, u, rowvar=False)[:m, m:])
df_r= pd.DataFrame(r,columns=['Correlation'])
df_r.to_csv("r.csv")

print("------>CERINTA 3------->")
r2 = r ** 2
x = 1 - r2
df = [(p - k + 1) * (q - k +1) for k in range(1, m+1)]
l = np.flip(np.cumprod(np.flip(x)))
chi2_val = (-n + 1 + (p + q + 1) / 2) * np.log(l)
p_values = 1 - chi2.cdf(chi2_val, df)

etichete_radacini = ["root" + str(i+1) for i in range(m)]
df_semnificatie = pd.DataFrame(
    {
        "R": r,
        "R2": r2,
        "p-value": p_values
    }
)

print(df_semnificatie)

nr_radacini = max(2, np.where(p_values > 0.01)[0][0])
print(nr_radacini)