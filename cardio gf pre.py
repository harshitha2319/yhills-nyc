
import numpy as np
import  pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


mydata=pd.read_csv("CardioGoodfitness.csv")

mydata.sample(10)

mydata.shape

mydata.dtypes

mydata.columns

mydata.info()

mydata.isnull().sum()

mydata.describe()

mydata.describe(include="all").T

sns.countplot(x="Product",data=mydata)

sns.countplot(x="Gender",data=mydata)

sns.countplot(x="MaritalStatus",data=mydata)

sns.countplot(x="Product",hue="Gender",data=mydata)

sns.countplot(x="Product",hue="MaritalStatus",data=mydata)

mydata.hist(figsize=(10,20))

plt.show()

sns.boxplot(x="Product",y="Age",data=mydata)

pd=crosstab(mydata[x="Product",y="Age",data=mydata)

sns.pairplot(mydata,diag_kind="kde")
                   
corr=mydata.corr()
corr
                   
sns.heatmap(corr,annot=true,(map="YlGnBu")
