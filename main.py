import pandas as pd 
import numpy as np 
import statsmodels.discrete.discrete_model as st 
from patsy import *
import pylab as plt 
import statsmodels.api as st
from sklearn.metrics import classification_report
from clean import prep

"""Data Preperation"""

y_train, x_train, y_test, x_test = prep('training.csv', 0.84)

model = st.Logit(y_train, x_train)
res = model.fit()
print res.summary() #useful to check if the predictors are significant or not, since p=val <0.05 for the chosen columns, it means they're all significant predictors

"""Model Evaluation using the training dataset"""
"""Distribution of predictions"""
kde_res =  st.nonparametric.KDEUnivariate(res.predict())
kde_res.fit()
plt.plot(kde_res.support, kde_res.density)
plt.fill_between(kde_res.support, kde_res.density, alpha = 0.5)
plt.title("Distribution of the predictions")

"""Change of survival probability by gender"""
plt.scatter(res.predict(), x_train['C(Sex)[T.male]'], alpha=0.5)
plt.grid(b=True, which='major', axis='x')
plt.xlabel('Predicted chance of survival')
plt.ylabel('Gender=> 1:male, 0:female')
plt.title('The change of survival probability by gender')

"""The Change of Survival Probability by class"""
plt.scatter(res.predict(),x_train['C(Pclass)[T.3]'] , alpha=0.2)
plt.xlabel("Predicted chance of survival")
plt.ylabel("Class Bool") # Boolean class to show if its 3rd class
plt.grid(b=True, which='major', axis='x')
plt.title("The Change of Survival Probability by class")

"""Change of survival probability by age"""
plt.scatter(res.predict(),x_train.Age , alpha=0.2)
plt.grid(True, linewidth=0.15)
plt.title("The Change of Survival Probability by Age")
plt.xlabel("Predicted chance of survival")
plt.ylabel("Age")

"""Evaluation based on test data """

y_pred = res.predict(x_test)
y_pred_flag = y_pred > 0.7

print pd.crosstab(y_test.Survived, y_pred_flag,rownames = ['Actual'],colnames = ['Predicted'])
print classification_report(y_test, y_pred_flag)




