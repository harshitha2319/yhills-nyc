#LOGISTIC REGRESSION

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/titanic-training-data.csv")
df.shape

df.describe()

df.dtypes

df.isnull().sum()

df.drop("Cabin",axis=1,inplace=True)

median1=df["Age"].median()
median1

df["Age"].replace(np.nan,median1,inplace=True)

model1=df["Embarked"].mode().values[0]
model1

df["Embarked"].replace(np.nan,median1,inplace=True)

df=pd.get_dummies(df,columns=['Sex'])
df.sample(10)


df=pd.get_dummies(df,columns=['Embarked'])
df.sample(10)

df=pd.get_dummies(df,columns=['Pclass'])
df.sample(10)

sns.pairplot(df,diag_kind="kde")

df.drop("Name",axis=1,inplace=True)

df.drop("Fare",axis=1,inplace=True)

df.drop("Ticket",axis=1,inplace=True)

df.drop("PassengerId",axis=1,inplace=True)

X=df.drop(["Survived"],axis=1)
Y=df[["Survived"]]


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.03)

df.dtypes


model_1=LogisticRegression()
model_1.fit(x_train,y_train)

model_1.score(X_train,Y_train)

model_1.score(X_test,Y_test)


from sklearn import metrics

predictions=model_1.predict(X_test)

print(metrics.classifications_report(Y_test,predictions))

from sklearn.metrics import  confusion_report
confusion_matrix(Y_test,predictions)

from sklearn.metrics import classification_report 
classification_report (Y_test,predictions)


cm=metrics.confusion_matrix(y_test,predction,labels=[1,0])
df_cm=pd.DataFrame(cm,index=[i for i in["1,0"]]
                  columns=[i for i in ["Predict 1","Predict 0"]])


plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=true,fmt='g')


from sklearn.tree import DecisionTreeClassifier

model_2=DecisionTreeClassifier()

model_2.fit(X_train,Y_train)

model_2.score(X_train,Y_train)

model_2.score(X_test,Y_test)

model_2_opt=DecisionTreeClassifier(max_depth=3)

model_2_opt.score(X_train,Y_train)

model_2_opt.score(X_test,Y_test)



from sklearn.svm import SVC

model_3=SVC()

model_3.fit(X_train,Y_train)

model_3.score(X_train,Y_train)

model_3.score(X_test,Y_test)

model_3_opt=SCV(kernel= "linear")

model_3.opt.fit(X_train,Y_train)

model_3.opt.score(X_train,Y_train)

model_3.opt.score(X_test,Y_test)




from sklearn.ensemble import BaggingClassifier

model_4=Baggingclassifier()

model_4.fit(X_train,Y_train)

model_4.score(X_train,Y_train)

model_4_opt=BaggingClassifier(base estimator=model_2_opt,n_estimators=50)

model_4.opt.fit(X_train,Y_train)

model_4.opt.fit(X_test,Y_test)

from sklearn.ensemble import AdaBoostClassifier()

model_5=AdaboostClassifier()

model_5.fit(X_train,Y_train)

model_5.score(X_train,Y_train)

from sklearn.ensemble import GradientBoostingClassifier

model_6=GradientBoostingClassifier()

model_6.fit(X_train,Y_train)

model_6.score(X_train,Y_train)

model_6.score(X_test,Y_test)

from sklearn.ensemble import RandomForestClassifier

model_7=RandomForestClassifier(max_features=5,n_estimator=25,max_depth=3)

model_7.fit(X_train,Y_train)

model_7.score(X_train,Y_train)

model_7.score(X_test,Y_test)

from sklearn.ensemble import BaggingRegressor
 
model_8=BaggingRegressor()

model_8.fit (X_train,Y_train)

model_8.score(X_train,Y_train)








