# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:51:42 2020

@author: SAMRAH SOHA
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

election=pd.read_csv(' E:/Assisgnments/Logistic Regression/election_data.csv')
election.columns="ID","Results","Years","Amount","Rank"
election.head()
election.info()
election.loc[election.Results>0,'Results']=1
#Removing first line as the record is empty
election=election.drop(election.index[[0]],axis=0)
election.describe()
# Model building 
#Election ID is not significant for election result hence, preparing model without Election ID
election=election.drop('ID',axis=1)
import statsmodels.formula.api as sm

Model2=sm.logit('Results~Years+Amount', data=election).fit()
Model2.summary()
Model2.summary2()  #AIC=15.80
Pred=Model2.predict(pd.DataFrame(election[['Years','Amount']]))
from sklearn.metrics import confusion_matrix
Confusion_Table=confusion_matrix(election['Results'],Pred>0.45)
Confusion_Table
Accuracy=9/10

