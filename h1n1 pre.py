import numpy as np
import pandas as pd

trian=pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/h1n1_vaccine_prediction.csv")
test = pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/h1n1_vaccine_prediction.csv")
submission = pd.read_csv("https://raw.githubusercontent.com/Premalatha-success/Datasets/main/h1n1_vaccine_prediction.csv")

train.head()
                  
train.info

train.isnull().sum()
          
                  
#!pip install autoviz
#!pip install xlrd
                  
from autoviz.Autoviz_Class import AutoViz_Class
import matplotlib.pyplot as plt
%matlplotlib inline
import seaborn as sns
 
AV=AutoViz_Class()
                  
filename = 'https://raw.githubusercontent.com/Premalatha-success/Datasets/main/h1n1_vaccine_prediction.csv'
sep = ","
dft = AV.AutoViz(
    filename,
    sep=",",
    depVar="",
    dfte=None,
    header=0,
    verbose=0,
    lowess=False,
    chart_format="svg"
)


           
