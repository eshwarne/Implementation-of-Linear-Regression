import pandas as pd
import numpy as np
from sklearn import cross_validation,svm,preprocessing
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('./testset.csv')
# print(df.head())

#extract important data alone
df.fillna(-9999,inplace=True)
df=df[[' _hum',' _tempm',' _pressurem',' _rain']]
df['label']=df[' _rain']
del df[' _rain']
print(df.head())
forecast_col=' rain'


#lets come to the machine learning part

x = np.array(df.drop('label',1)) #use numpy array
x=preprocessing.scale(x)
y = np.array(df['label'])
print(len(x),len(y))
x_train, x_test, y_train, y_test=cross_validation.train_test_split(x,y,test_size=0.8)
# clf=LinearRegression()
# clf.fit(x_train,y_train)

#use pickling so that we do not have to train again and again
# with open('mylinearregression.pickle','wb') as fp:
#     pickle.dump(clf,fp)
pickle_in=open('mylinearregression.pickle','rb')
clf=pickle.load(pickle_in)
acc=clf.score(x_test,y_test)
print(acc)
# x = preprocessing.scale(x)