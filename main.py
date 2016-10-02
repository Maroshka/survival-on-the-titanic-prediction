import pandas as pd 
import numpy as np 
import clean as cl 

"""Data Preperation"""
df = pd.read_csv('training.csv')
#removing cols that dont add much value to the data
del df['Ticket']
del df['Cabin']
del df['Name']
#removing missing values
df = df.dropna()
#deviding the dataset into training and testing datasets
dftrain = df[:600 , : ]
dftest = df[600: , : ]
print df 

