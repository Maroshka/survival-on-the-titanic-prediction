# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 03:51:47 2016

@author: muna
"""
import pandas as pd
import pylab as plt
import numpy as np

""" Plotting the no. of survivors in each class """
data = pd.read_csv('training.csv')
#print data.columns.values #print the colmn names 
#print data['Survived'].isnull().value_counts()
#print data['Pclass'].isnull().value_counts()
survivors = data.groupby('Pclass')['Survived'].agg(sum)
total_passengers = data.groupby('Pclass')['PassengerId'].count()
survivalPercentage = survivors/total_passengers
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(survivors.index.values.tolist(), survivors, color ='green', width=0.3)
ax.set_xlabel('class')
ax.set_ylabel('No. of survivors')
ax.set_title('Total no. of survivors based on class')
xtickMarks = survivors.index.values.tolist()
ax.set_xticks(survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xtickMarks)
#plt.setp(xtickNames, fontsize=15)
plt.show()

""" Plotting the percentage of survivors in each class """
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
rect2 = ax2.bar(survivalPercentage.index.values.tolist(), survivalPercentage, color='yellow', 
width = 0.3)
ax2.set_ylabel('percentage of survival')
ax2.set_title('Percentageof survival based on class')
xtickMarks2 = survivalPercentage.index.values.tolist()
ax2.set_xticks(survivalPercentage.index.values.tolist())
xtickNames2 = ax.set_xticklabels(xtickMarks2)
plt.setp(xtickNames2, fontsize = 15)
plt.show()

""" Distribution of survivors based on gender among different classes """
msurvivors = data[data['Sex']=='male'].groupby('Pclass')['Survived'].agg(sum)
wsurvivors = data[data['Sex']=='female'].groupby('Pclass')['Survived'].agg(sum)

index = np.arange(msurvivors.count())
fig3 = plt.figure()
ax2 = fig3.add_subplot(111)
rect1 = ax2.bar(index, msurvivors, color='blue', 
width = 0.3, label = 'Men')
rect2 = ax2.bar(index+0.3, wsurvivors, color='y', width = 0.3, label='Women')
ax2.set_ylabel("male survivors")
ax2.set_title('male and female survivors based on class')
xtickMarks2 = msurvivors.index.values.tolist()
ax2.set_xticks(index+0.3)
xtickNames2 = ax.set_xticklabels(xtickMarks2)
plt.setp(xtickNames2, fontsize = 15)
plt.legend()
plt.tight_layout()
plt.show()

""" Percentage of male and femal survivals based on class """
totMs = data[data['Sex']=='male'].groupby('Pclass')['PassengerId'].count()
totFs = data[data['Sex']=='female'].groupby('Pclass')['PassengerId'].count()
mpsurvivors = msurvivors/totMs
wpsurvivors = wsurvivors/totFs
index = np.arange(msurvivors.count())
fig4 = plt.figure()
ax2 = fig4.add_subplot(111)
rect1 = ax2.bar(index, mpsurvivors, color='blue', 
width = 0.3, label = 'Men')
rect2 = ax2.bar(index+0.3, wpsurvivors, color='y', width = 0.3, label='Women')
ax2.set_ylabel("No. of survivors")
ax2.set_title('Percentage of male and female survivors based on class')
xtickMarks2 = msurvivors.index.values.tolist()
ax2.set_xticks(index+0.3)
xtickNames2 = ax.set_xticklabels(xtickMarks2)
plt.setp(xtickNames2, fontsize = 15)
plt.legend()
plt.tight_layout()
plt.show()

"""Distribution of nonsurvivors based on class who have family abroad the ship"""
nonsrv = data[(data['Parch'] > 0) | (data['SibSp'] > 0) & (data['Survived']==0)].groupby('Pclass')['Survived'].agg('count')
x = nonsrv.index.values.tolist()
fig5 = plt.figure()
ax2 = fig5.add_subplot(111)
rect1 = ax2.bar(x, nonsrv, color='red', width = 0.3)
ax2.set_ylabel("No. of nonsurvivors")
ax2.set_title('Distro of nonsrv with family abroad based on class')
ax2.set_xticks(x)
xtickNames2 = ax.set_xticklabels(x)
plt.setp(xtickNames2, fontsize = 15)
plt.show()

""" Percentage of the prev analysis """
totno = data.groupby('Pclass')['PassengerId'].count()
perc = nonsrv/totno
x = nonsrv.index.values.tolist()
fig6 = plt.figure()
ax2 = fig6.add_subplot(111)
rect1 = ax2.bar(x, perc, color='red', width = 0.3)
ax2.set_ylabel("Percentage of nonsurvivors")
ax2.set_title('Percentage of nonsurvivors with family abroad based on class')
ax2.set_xticks(x)
xtickNames2 = ax.set_xticklabels(x)
plt.setp(xtickNames2, fontsize = 15)
plt.show()