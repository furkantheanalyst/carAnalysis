# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 14:05:02 2021

@author: Furkan
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


#Loading Dataset
dataFrame=pd.read_excel("mercedes.xlsx")
#Checking dataset and correlation between variables
print(dataFrame.head())
print(dataFrame.describe())
print(dataFrame.isnull().sum().sort_values(ascending=False))
print(dataFrame.corr())

#Analysis of Models
sns.set_style("white")
plt.figure(figsize=(20,10))
ax = sns.countplot(x="model", data=dataFrame)
plt.xticks(rotation=90)
plt.title("Models By Percentage", fontsize=20)

total = len(dataFrame)

for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width() / 2.
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center',va='bottom')
plt.show()

#Analysis of Price And Year
plt.figure(figsize=(20, 10))
sns.scatterplot(x='year', y="price", data=dataFrame)
plt.xticks(rotation=90)
plt.show()

#Analysis of Price and Transmission
plt.figure(figsize=(20, 10))
sns.catplot(x="transmission", y="price", kind="boxen", data=dataFrame.sort_values("price",ascending=False), aspect=3)
plt.show()

#Analysis of Price and FuelType
plt.figure(figsize=(20, 10))
sns.catplot(x="fuelType", y="price", kind="boxen", data=dataFrame.sort_values("price",ascending=False), aspect=3)
plt.show()

#Analysis of Price and Continuous Variables
sns.scatterplot(x="mileage",y="price",data=dataFrame)
plt.show()
sns.scatterplot(x="tax",y="price",data=dataFrame)
plt.show()
sns.scatterplot(x="mpg",y="price",data=dataFrame)
plt.show()
sns.scatterplot(x="engineSize",y="price",data=dataFrame)
plt.show()

#There is some outliers in our dataset. I will drop them.
dataFrame=dataFrame[dataFrame.year !=1970]
#Dropping 0.01 of our data. Outliers.
dataFrame = dataFrame.sort_values('price',ascending = False).iloc[int(len(dataFrame) * 0.01):]

#Converting Categorical Variables To Dummies
dataFrame = pd.get_dummies(dataFrame, columns=['model','transmission','fuelType'])
#Dropping trap variables
dataFrame.drop(['model_230','transmission_Other','fuelType_Other'],axis=1,inplace=True)

#Model Building
# Separate Dependent and Independent Variables
X = dataFrame.drop('price',axis=1)
y = dataFrame['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)
# print(len(X_train))
# print(len(X_test))
# print(len(y_train))
# print(len(y_test))

#We will scale the variables for easier process
scaler=MinMaxScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

model=Sequential()
model.add(Dense(12,activation="relu"))

model.add(Dense(24,activation="relu"))
model.add(Dense(24,activation="relu"))
model.add(Dense(24,activation="relu"))

model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")

model.fit(x=X_train, y = y_train,validation_data=(X_test,y_test),batch_size=250,epochs=300)

#Checking model performance
loss=pd.DataFrame(model.history.history)
loss.plot()
plt.show()
print(loss.head())

predictions= model.predict(X_test)
print("\nmean absolute error:", mean_absolute_error(y_test,predictions))

plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,"g-*")
plt.show()

#Crosscheck
print(dataFrame.iloc[10])
newCarSeries = dataFrame.drop("price",axis=1).iloc[2]
newCarSeries = scaler.transform(newCarSeries.values.reshape(-1,37))
print(model.predict(newCarSeries))