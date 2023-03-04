#*/linear regression/*



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns 
from sklearn.model_selection import trin_test_split
from sklearn.linear_model import LinearRegression


car_df=pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Dataset/main/auto-mpg.csv")

car_df.shape

car_df.sample(10)

car_df.drop("car name",axis=1,inlpace=True)

X=car_df.drop(['mpg'],axis=1)
y=car_df[['mpg']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30)
model_1=LinearRegression()
model.1.fit(X_trian,y_train)
model_1.score(X_train,,y_train)
model_1.score(X_test,y_test)


from sklearrn.preprocessing import PolynomialFeatures
from sklearn import linear_model 

car_df['origin'].replace({1:'america',2:'europe',3:'asia'})
car_df.sample(10)

car_df=pd.get_dummies(df,columns=['origin'])
car_df.sample(10)

car_df.info()

car_df.isnull()sum()

car_df.descrbe(include="all")

car_df['horsepower']=df.[horsepower'].replace("?",np.nan)

car_df.dtypes

car_df.head()

dup=car_df.duplicated()
print(dup.sum())

car_df.duplicated()sum()


def rem_out(col):
    sorted(col)
    q1,q3=col.quantile([0.25,0.75])
    iqr=q3-q1
    l_ran=q1-1.5*iqr
    u_ran=q1+1.5*iqr
    return l_ran,u_ran
   
u_lt,h_lt=rem_out(d['mpg])
car_df['mpg']=np.where(car_df['mpg']>h_lt,h_lt,car_df.['mpg'])
car_df.['mpg']=np.where(car_df['mpg']>u_lt,u_lt,car_df.['mpg'])


car_df.boxplot(column.['mpg'])
car_df.boxplot(column.['cylinders'])
car_df
car_df.isna().sum()
columns=['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin']
dummies=pd.get_dummies(car_df.[columns])
car_df=pd.concat([car_df,dummies],axis=1)
