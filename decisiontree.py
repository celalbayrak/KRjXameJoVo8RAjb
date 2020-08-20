# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 19:39:13 2020

@author: Celal
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
#%%
df=pd.read_csv("term-deposit-marketing-2020.csv")
bins = [0, 10, 20, np.inf]
names = ['0-10', '10-20', '20-31']
df['day'] = pd.cut(df['day'], bins, labels=names) #CATEGORIZE DAY COLUMN
#%%
df_dum= pd.get_dummies(df) #ENCODE CATEGORICAL COLUMNS
#%% 
df_dum_final=df_dum.drop(['default_no','housing_no','loan_no','y_no'],axis=1) #DROP REDUNDANT COLUMNS
#%%
x=df_dum_final.drop("y_yes",axis=1) #SEPERATE LABELS AND FEATURES
y=df_dum_final["y_yes"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#%%
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train) #CALCULATE CLASS WEIGHTS
#%%
def run_gridsearch(X, y, clf, param_grid, cv=5):  #GRID SEARCH FOR BEST PARAMETERS, WITH 5-FOLD CROSS VALIDATION
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv, scoring="roc_auc", verbose=1,n_jobs=-1)
    grid_search.fit(X, y)
    print("roc auc: ", grid_search.best_score_)
    print("Best Params: ", grid_search.best_params_)
    return 1
#%%
dt_temp=DecisionTreeClassifier(class_weight={0:class_weights[0],1:class_weights[1]})
params= {"max_depth":[3,4,5,6,7,8,9,10,12,14,16,18,20],"criterion":["gini","entropy"],  #CANDIDATE PARAMETERS
         "min_samples_split":[2,3,4,5,10,20,40,60,80,100,120,200,250,300,350,400,500],
         "min_samples_leaf":[2,3,4,5,6,10,20,40,80,160]}
run_gridsearch(x_train,y_train,dt_temp,params,cv=5)
#%%
dt=DecisionTreeClassifier(class_weight={0:class_weights[0],1:class_weights[1]},     #BEST PARAMETERS, FROM GRIDSEARCH
                                        max_depth=12,criterion="gini",
                                        min_samples_split=350,min_samples_leaf=20)
dt.fit(x_train,y_train)
conf_mat=confusion_matrix(y_test,dt.predict(x_test))    #CONFUSION MATRIX OF BEST MODEL ON TEST DATA
#%%
scores = cross_val_score(dt, x_train, y_train, cv=5, scoring='accuracy')#MEASURING THE ACCURACY BY EVALUATING 5-FOLD CROSS VALIDATION
avg_acc=np.mean(scores)
print("Average cross validation accuracy: "+ str(avg_acc))
#%%

print("Test roc auc score: "+str(roc_auc_score(y_test,dt.predict(x_test)))) #PRINT TEST SCORES
print("Test accuracy score: "+str(accuracy_score(y_test,dt.predict(x_test))))
print("Test recall score: "+str(recall_score(y_test,dt.predict(x_test))))
#%%
print(dict(sorted(zip(dt.feature_importances_,x.columns)))) #LIST THE FEATURE IMPORTANCES
#%%
from dtreeviz.trees import *
viz = dtreeviz(dt,                  #VISUALIZE THE DECISION TREE
               x_train,
               y_train,
               target_name='TARGET',
               feature_names=x_train.columns, 
               class_names=["NO","YES"], 
               histtype= 'barstacked')
viz.save("decision_tree.svg")
