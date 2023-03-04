import numpy as np 
import pandas as pd

import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as MSE


train = pd.read_csv("train.csv", nrows = 1000000)(****###I used (https://raw.githubusercontent.com/Premalatha-success/Datasets/main/TaxiFare.csv))
test = pd.read_csv("test.csv")

train.head()

test.head()

train.shape,test.shape

train.isnull().sum()

train = train.dropna(how ='any', axis ='rows') 

train = train.dropna(how ='any', axis ='rows') 

train.info()

train['fare_amount'].describe()

train.drop(train[train['passenger_count'] == 0].index, axis=0, inplace = True)

train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)


train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour
train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.day_name()
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.day
train.loc[:, 'pickup_month'] = train['pickup_datetime'].dt.month
train.loc[:, 'pickup_day'] = train['pickup_datetime'].dt.dayofweek
test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour
test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.day_name()
test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.day
test.loc[:, 'pickup_month'] = test['pickup_datetime'].dt.month
test.loc[:, 'pickup_day'] = test['pickup_datetime'].dt.dayofweek



train.drop(['key','pickup_datetime'], axis=1,inplace=True)

train.dropna(inplace=True)

train.head()

train = train.drop(['pickup_weekday'],axis=1)

x, y = train.drop('fare_amount', axis = 1), train['fare_amount']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)


scaler = StandardScaler()
scaler.fit_transform(x_train,x_test)

xgb_r = xgb.XGBRegressor(objective ='reg:linear',n_estimators = 400, seed = 123)

xgb_r.fit(x_train,y_train)


y_pred = xgb_r.predict(x_test)


rmse = np.sqrt(MSE(y_test, y_pred))
rmse


test.drop(['key','pickup_datetime'], axis=1,inplace=True)



test.dropna(inplace=True)

test.drop(test.index[(test.pickup_longitude < -75) | 
           (test.pickup_longitude > -72) | 
           (test.pickup_latitude < 40) | 
           (test.pickup_latitude > 42)],inplace=True)
test.drop(test.index[(test.dropoff_longitude < -75) | 
           (test.dropoff_longitude > -72) | 
           (test.dropoff_latitude < 40) | 
           (test.dropoff_latitude > 42)],inplace=True)
           
           
test.drop(['pickup_weekday'], axis=1,inplace=True)


scaler.fit_transform(test)


new_pred = xgb_r.predict(test)


sample_new=pd.read_csv('test.csv')

sample_new.drop(['pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count'], axis=1,inplace=True)


sample_new['fare_amount'] = new_pred


sample_new.head()


submission=sample_new.to_csv("submission.csv", index=False)
