import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
from sklearn.preprocessing import LabelEncoder


#Loaidng the datasets
df = pd.read_csv('Loan_defualter.csv')

#droping the unwanted columns
df.drop(['RowNumber', 'CustomerId','Surname'],axis=1,inplace=True)

#Categorical data encoding
le = LabelEncoder()
objList = df.select_dtypes(include = "object").columns
print (objList)

for feat in objList:
    df[feat] = le.fit_transform(df[feat].astype(str))
print (df.info())

#Splitting the data into two variables
X = df.drop(['Exited'],axis=1)
y = df['Exited']

#splitting the data in training anf testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#training/fitting the model on training dataset
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)

#predictig the values on testing dataset
predictrfc = rfc.predict(X_test)

#saving the model
pickle.dump(rfc,open('predictionrfc.pkl','wb'))
