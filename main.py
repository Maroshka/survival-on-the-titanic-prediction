import pandas as pd 
import numpy as np 
import clean as cl 
import statsmodels.discrete.discrete_model as st 
from patsy import *

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

#y_train, x_train = dftrain['Survived'], dftrain[['Age', 'Sex', 'Parch', 'SibSp', 'Pclass', 'Embarked']]
#y_test, x_test = dftest['Survived'], dftest[['Age', 'Sex', 'Parch', 'SibSp', 'Pclass', 'Embarked']]

y_train,x_train = dmatrices(formula, data=dftrain,return_type='dataframe')

model = st.Logit(y_train, x_train)
res = model.fit()
print res.summary()




