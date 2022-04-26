import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df= pd.read_csv("AQI Data.csv")

df.isnull().sum()

df=df.dropna()

df.isnull().sum()

x=df.iloc[:,:-1] ## independent features
y=df.iloc[:,-1] ## dependent features

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(x,y)

from sklearn.preprocessing import StandardScaler
scale= StandardScaler()
x= scale.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


#Linear Regression algorithm
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_train,y_train)

prediction=regressor.predict(X_test)

from sklearn.metrics import r2_score
print("r2_score is",r2_score(y_test, prediction))

#XGBOOST algorithm

from xgboost import XGBRegressor

xg= XGBRegressor()
xg.fit(X_train,y_train)
pred=xg.predict(X_test)
print("r2_score is",r2_score(y_test, pred))

#Random Forest algorithm

from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor()
rf.fit(X_train,y_train)
pre=rf.predict(X_test)
print("r2_score is",r2_score(y_test, pre))
 
#Pickle file

import pickle
filename = 'air_model.pkl'
pickle.dump(xg, open(filename, 'wb'))