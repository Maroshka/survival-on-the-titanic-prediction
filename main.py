import pandas as pd 
import numpy as np 
import clean as cl 
import statsmodels.discrete.discrete_model as st 
from patsy import *
import pylab as plt 
import statsmodels.api as st
from sklearn.metrics import classification_report

"""Data Preperation"""
df = pd.read_csv('training.csv')
#removing cols that dont add much value to the data
del df['Ticket']
del df['Cabin']
del df['Name']
#removing missing values
df = df.dropna()
#deviding the dataset into training and testing datasets
dftrain = df[:600]
dftest = df[600:]

formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp'
f2 = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp'

#y_train, x_train = dftrain['Survived'], dftrain[['Age', 'Sex', 'Parch', 'SibSp', 'Pclass', 'Embarked']]
#y_test, x_test = dftest['Survived'], dftest[['Age', 'Sex', 'Parch', 'SibSp', 'Pclass', 'Embarked']]

y_train,x_train = dmatrices(formula, data=dftrain,return_type='dataframe')
y_test, x_test = dmatrices(f2, data=dftest,return_type='dataframe')

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




