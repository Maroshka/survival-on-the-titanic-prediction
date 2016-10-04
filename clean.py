
import pandas as pd 
import numpy as np 
from math import *
from patsy import *

"""Data Preperation"""
def prep(filename,ratio):

	df = pd.read_csv(filename)
	#removing missing values
	df = df.dropna()
	#deviding the dataset into training and testing datasets
	n = int(floor(len(df)*ratio))
	dftrain = df[:n]
	dftest = df[n:]
	formula = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp'
	y_train,x_train = dmatrices(formula, data=dftrain,return_type='dataframe')
	y_test, x_test = dmatrices(formula, data=dftest,return_type='dataframe')
	
	return y_train, x_train, y_test, x_test