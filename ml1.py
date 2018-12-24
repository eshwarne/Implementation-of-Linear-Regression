#Regression 
#Features and Labels
#Features -- attributes
#Labels -- determines the prediction

import pandas as pd 
import quandl
import math
import numpy as np 
from sklearn import preprocessing, cross_validation,  svm
from sklearn.linear_model import LinearRegression
from matplotlib import style
import matplotlib.pyplot as plt 
style.use('ggplot')
df=quandl.get('WIKI/GOOGL') #for getting data

forecast_col='Adj. Close'

#In machine learning you cannot work with NaN data,
#so we replace nan with some value value using
#dataframe.fillna function which is available in pandas
df.fillna(-9999,inplace=True)


#use data that came ten days ago to predict today kinda
forecast_out = int(math.ceil(0.1*len(df)))

#shift rows upwards
df['label'] = df[forecast_col].shift(-forecast_out)
# print(df.head())



#### THE MACHINE LEARNING PART ######
x = np.array(df.drop(['label'])) #use numpy array
y = np.array(df['label'])

x = preprocessing.scale(x)

#we shifted forecast_out, say 10 rows up, so there would be 10 empty rows
#remove them!
x = x[:-forecast_out+1] 
df.dropna(inplace=True)
y = np.array(df['label'])
print(len(x),len(y))

#decide data for training and data for testing
#cross validation shuffles rows
x_train, X_test, y_train, Y_test=cross_validation.train_test_split(x,y,test_size=0.2)
#we can change classifiers
#clf=svm.SVR();
clf=LinearRegression()

#fit - train
#score - test
clf.fit(x_train,y_train)
accuracy = clf.score(X_test,Y_test)

#n_jobs = threading default value is 1
#n_jobs = -1 means using all cpus


