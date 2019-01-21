import pandas as pd 
from sklearn.preprocessing import Imputer, StandardScaler
base = pd.read_csv('credit-data.csv')
print(base.loc[base['age']<0])

#base.drop('age',1,inplace=True) apagar a coluna 'age'
#base.drop(base[base.age < 0].index, inplace = True)
#print(base.loc[base['age']<0])

#preencer os valores com a media 

media = base['age'].mean()


#media dos valores maiores que zero
#base['age'][base.age>0].mean()
base.loc[base.age<0] = media 

pd.isnull(base['age'])  # procura por valores nulos dentro de age 
base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:,1:4].values
classificadores = base.iloc[:4].values

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)  ## substitui valoes por media | mediana | mais frequente 
imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

scaler = StandardScaler() ## colocar dados na mesma escala

previsores = scaler.fit_transform(previsores)
print(previsores)