import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import hstack
from numpy import vstack
from numpy import asarray
import warnings
warnings.filterwarnings('ignore') 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
#from imblearn.over_sampling import SMOTE, RandomOverSampler
#from imblearn.under_sampling import NearMiss, RandomUnderSampler
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Tuple, Any, Union, Optional, Dict




# build gridsearch
def build_grid_search_pipline(clf_model, clf_hyper_params, scoring='f1', cv=None, n_jobs=-1, error_score=0, verbose=0):
    
    pipe = Pipeline(
        steps=[
            ('clf', clf_model),
        ]
    )
    cv = ShuffleSplit(test_size=0.2, n_splits=1)
    return GridSearchCV(pipe, clf_hyper_params, scoring=scoring, cv=cv, n_jobs=n_jobs, error_score=error_score, verbose=verbose)

def get_hyper_params(model_name):
    hyper_params_for_all_models ={
        'LogisticRegression':{ 
            'clf__class_weight': [None, 'balanced'],
            # 'clf__C':  [1,2,3,5,10,20],
            # 'clf__max_iter': [100, 500],
            # 'clf__penalty' : ['none',  'l2'],
            # 'clf__solver' : ['newton-cg', 'lbfgs', 'liblinear',  'sag', 'saga']
        },
        'SGDClassifier' :{
            "clf__loss":  ['log'],
            # "clf__penalty": ['l1','l2','elasticnet'],
            # "clf__max_iter": [5,10,100],
            # "clf__alpha": [1, 1e-3, 1e-2, 1e-1],
        },
        'SVC':{
            'clf__C': [0.1, 1, 2],#10, 20],
            # 'clf__gamma': [1, 0.1, 0.01, 0.001],
            # 'clf__kernel': ['poly', 'sigmoid']
        },
        'MultinomialNB': {
            "clf__alpha" :  np.linspace(0.7, 1.5, 3),
            # "clf__fit_prior" : [True,False],
            # "clf__alpha" : [1, 0.01, 0.1, 0.2, 0.3, 0.4],
            # "clf__alpha" : [1, 0.01, 0.1, 0.2],
        },
        'GradientBoostingClassifier' :{
            "clf__loss":["deviance"],
            # "clf__learning_rate": [0.01,  0.05, 0.1, 0.15, 0.2],
            # "clf__min_samples_split": np.linspace(0.1, 0.5, 8),
            # "clf__min_samples_leaf": np.linspace(0.1, 0.5, 8),
            # "clf__max_depth":[3,5,8],
            # "clf__max_features":["log2","sqrt"],
            # "clf__criterion": ["friedman_mse",  "mae"],
            # "clf__subsample":[0.6, 0.8, 1.0],
            "clf__n_estimators":[10]
        },
        'RandomForestClassifier': {
            'clf__n_estimators' : [5, 10, 20], 
            # 'clf__n_estimators' : [10, 50, 100], 
            # 'clf__max_depth' : [5, 8, 15, 25],
            # 'clf__min_samples_split' : [8, 15, 50],
            # 'clf__min_samples_leaf' : [1, 2, 5, 10], 
        },
        'AdaBoostClassifier' :{
            'clf__n_estimators' : [10, 50, 100], 
            # l'clf__learning_rate':[.001,0.01,.1]
        },
        'BaggingClassifier' :{
            'clf__n_estimators' : [10, 50, 100], 
            # 'clf__n_estimators' : [10, 20, 50], 
            # 'clf__base_estimator__max_leaf_nodes':[5, 10, 20], 
            # 'clf__base_estimator__max_depth':[5, 10 , 20],
        },
        'ExtraTreesClassifier' :{
            'clf__n_estimators' : [10, 50, 100], 
            # 'clf__n_estimators' : [10, 20, 50], 
        },
        'MLPClassifier':{
            # 'clf__hidden_layer_sizes': [(5,10,20), (10,20,30), (30,)],
            # 'clf__activation': ['tanh', 'relu'],
            # 'clf__solver': ['sgd', 'adam'],
            # 'clf__alpha': [0.0001, 0.05],
            'clf__learning_rate': ['constant','adaptive'],
        },
        'KNeighborsClassifier' :{
                # 'clf__leaf_size' : list(range(1,20,2)),
                # 'clf__n_neighbors' : list(range(1,20,4)), 
                'clf__p' : [1,2],
        },
        'DecisionTreeClassifier' :{
            'clf__criterion':['gini','entropy'],
            # 'clf__max_depth':[5, 10 , 20],
        },
        'XGBClassifier': {
            'clf__n_estimators' : [10, 50, 100], 
            # 'clf__subsample': [0.9, 1],
            # 'clf__alpha': [1.5, 2, 2.5],
            # 'clf__learning_rate': [0.01, 0.1,0.2,0.3],
            # 'clf__gamma': [i/10.0 for i in range(0,4)],
            # 'clf__booster':['gbtree','dart'],
        },
    }
    
    scoring = make_scorer(f1_score, average = 'binary')
    return hyper_params_for_all_models[model_name], scoring

def get_gridsearchcv_obj(model):
    hyper_params, scoring = get_hyper_params(model.__class__.__name__)
    clf = build_grid_search_pipline(model, hyper_params, scoring)
    return clf