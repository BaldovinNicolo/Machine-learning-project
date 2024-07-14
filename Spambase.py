#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 19:45:51 2023

@author: nicolobaldovin
@matricola: 892115
"""

# First, we will start by importing the necessary libraries and loading the data 
from ucimlrepo import fetch_ucirepo 
import numpy as np
import random
import pandas as pd
from sklearn.svm import SVC
import tracemalloc
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
import time
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import norm 
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# Fetch dataset 
spambase = fetch_ucirepo(id=94) 


# DATASET
# link: "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"


# Split the data into features and labels
X = spambase.data.features 
y = spambase.data.targets 

# Drop features 55-57 starting counting from 1 (columns 54-56 starting from 0)
X = X.drop(columns=X.columns[[54, 55, 56]])
data = X + y

# Next, we will transform the data using the TF/IDF representation
X = TfidfTransformer().fit_transform(X)

omnia = {}

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1, stratify=np.ravel(y))

def SVM(X,y):
    start = time.time()
    tracemalloc.start()
    svm_linear = SVC(kernel="linear", C=1.0, random_state = 5)  # SVM with linear kernel
    results_linear = cross_val_score(svm_linear, X, np.ravel(y), cv=10)
    non, peak = tracemalloc.get_traced_memory()
    end = time.time()
    tracemalloc.stop()
    omnia["svm_linear"] = [end-start]
    omnia["svm_linear"].append(results_linear.mean())
    omnia["svm_linear"].append(results_linear.std())
    omnia["svm_linear"].append(int(4601 - results_linear.mean()*4601))
    omnia["svm_linear"].append(peak)
    print("svm_linear:")
    print('Time: %f'%(omnia["svm_linear"][0]))
    print('Missclassified examples: %d'% (omnia["svm_linear"][3]))
    print('Score: %.3f' %(omnia["svm_linear"][1]))
    print('Std: %0.3f' %(omnia["svm_linear"][2]))
    print('Memory: %d'% (omnia["svm_linear"][4]))
        
    start = time.time()
    tracemalloc.start()
    svm_poly = SVC(kernel="poly", degree=2, random_state = 5, C=1000.0) # SVM with polynomial kernel of degree 2
    results_poly = cross_val_score(svm_poly, X, np.ravel(y), cv=10)
    non, peak = tracemalloc.get_traced_memory()
    end = time.time()
    tracemalloc.stop()
    omnia["svm_poly"] = [end-start]
    omnia["svm_poly"].append(results_poly.mean())
    omnia["svm_poly"].append(results_poly.std())
    omnia["svm_poly"].append(int(4601 - results_poly.mean()*4601))
    omnia["svm_poly"].append(peak)
    print("\nsvm_poly:")
    print('Time: %f'%(omnia["svm_poly"][0]))
    print('Missclassified examples: %d'% (omnia["svm_poly"][3]))
    print('Score: %.3f' %(omnia["svm_poly"][1]))
    print('Std: %0.3f' %(omnia["svm_poly"][2]))
    print('Memory: %d'% (omnia["svm_poly"][4]))
    
    start = time.time()
    tracemalloc.start()
    svm_rbf = SVC(kernel="rbf", random_state = 5, C = 10.0) # SVM with RBF kernel
    results_rbf = cross_val_score(svm_rbf, X, np.ravel(y), cv=10)
    non, peak = tracemalloc.get_traced_memory()
    end = time.time()
    tracemalloc.stop()
    omnia["svm_rbf"] = [end-start]
    omnia["svm_rbf"].append(results_rbf.mean())
    omnia["svm_rbf"].append(results_rbf.std())
    omnia["svm_rbf"].append(int(4601 - results_rbf.mean()*4601))
    omnia["svm_rbf"].append(peak)
    print("\nsvm_rbf:")
    print('Time: %f'%(omnia["svm_rbf"][0]))
    print('Missclassified examples: %d'% (omnia["svm_rbf"][3]))
    print('Score: %.3f' %(omnia["svm_rbf"][1]))
    print('Std: %0.3f' %(omnia["svm_rbf"][2]))
    print('Memory: %d'% (omnia["svm_rbf"][4]))
    
# -------------------------------------------------------------
 
def angular_linear(X_train, y_train, X_test, y_test):                           
                                                                     
                                                                                
    kernel_matrix = cosine_similarity(X_train)
    kernel_tester = cosine_similarity(X_test, X_train)
    svm = SVC(kernel='precomputed')
    svm.fit(kernel_matrix, np.ravel(y_train))
    y_pred_svm = svm.predict(kernel_tester)
    # print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_svm))
    
def angular_polynomial(X_train, y_train, X_test, y_test):
    kernel_matrix = (cosine_similarity(X_train)+1)**2
    kernel_tester = (cosine_similarity(X_test, X_train)+1)**2
    svm = SVC(kernel='precomputed')
    svm.fit(kernel_matrix, np.ravel(y_train))
    y_pred_svm = svm.predict(kernel_tester)
    # print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_svm) + "\n")
    
# -------------------------------------------------------

# Random forests
def Random_forests(X,y):
    start = time.time()
    tracemalloc.start()
    random_forests = RandomForestClassifier(n_estimators=25, random_state = 1, n_jobs=2)
    results_randomforests = cross_val_score(random_forests, X, np.ravel(y), cv=10)
    non, peak = tracemalloc.get_traced_memory()
    end = time.time()
    tracemalloc.stop()
    omnia["random_forest"] = [end-start]
    omnia["random_forest"].append(results_randomforests.mean())
    omnia["random_forest"].append(results_randomforests.std())
    omnia["random_forest"].append(int(4601 - results_randomforests.mean()*4601))
    omnia["random_forest"].append(peak)
    print("\nrandom_forest:")
    print('Time: %f'%(omnia["random_forest"][0]))
    print('Missclassified examples: %d'% (omnia["random_forest"][3]))
    print('Score: %.3f' %(omnia["random_forest"][1]))
    print('Std: %0.3f' %(omnia["random_forest"][2]))
    print('Memory: %d'% (omnia["random_forest"][4]))
    
# k-NN with k=5    
def k_NN(X,y):
    start = time.time()
    tracemalloc.start()
    k_nn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    results_k_nn = cross_val_score(k_nn, X, np.ravel(y), cv=10)
    non, peak = tracemalloc.get_traced_memory()
    end = time.time()
    tracemalloc.stop()
    omnia["k_nn"] = [end-start]
    omnia["k_nn"].append(results_k_nn.mean())
    omnia["k_nn"].append(results_k_nn.std())
    omnia["k_nn"].append(int(4601 - results_k_nn.mean()*4601))
    omnia["k_nn"].append(peak)
    print("\nk_nn:")
    print('Time: %f'%(omnia["k_nn"][0]))
    print('Missclassified examples: %d'% (omnia["k_nn"][3]))
    print('Score: %.3f' %(omnia["k_nn"][1]))
    print('Std: %0.3f' %(omnia["k_nn"][2]))
    print('Memory: %d'% (omnia["k_nn"][4]))
    
# Naive Bayes classifier
# Convert the sparse matrix to a dense numpy array
def Naive_Bayes_Gauss(X,y):
    start = time.time()
    tracemalloc.start()
    X = X.toarray()                                 
    naive_bayes_gauss = GaussianNB()
    results_naive_bayes_gauss = cross_val_score(naive_bayes_gauss, X, np.ravel(y), cv=10)
    non, peak = tracemalloc.get_traced_memory()
    end = time.time()
    tracemalloc.stop()
    omnia["naive_bayes_gauss"] = [end-start]
    omnia["naive_bayes_gauss"].append(results_naive_bayes_gauss.mean())
    omnia["naive_bayes_gauss"].append(results_naive_bayes_gauss.std())
    omnia["naive_bayes_gauss"].append(int(4601 - results_naive_bayes_gauss.mean()*4601))
    omnia["naive_bayes_gauss"].append(peak)
    print("\nnaive_bayes_gauss:")
    print('Time: %f'%(omnia["naive_bayes_gauss"][0]))
    print('Missclassified examples: %d'% (omnia["naive_bayes_gauss"][3]))
    print('Score: %.3f' %(omnia["naive_bayes_gauss"][1]))
    print('Std: %0.3f' %(omnia["naive_bayes_gauss"][2]))
    print('Memory: %d'% (omnia["naive_bayes_gauss"][4]))
   
angular_linear(X_train, y_train, X_test, y_test)   
angular_polynomial(X_train, y_train, X_test, y_test) 
SVM(X,y)
Random_forests(X,y)
k_NN(X,y)
Naive_Bayes_Gauss(X,y)

support = ["svm_linear", "svm_poly", 'svm_rbf', 'random_forest', 'k_nn', 'naive_bayes_gauss']
plt.bar(support, [x[1] for x in omnia.values()], yerr = [x[2] for x in omnia.values()], align = "center", color="purple")

fig, axes = plt.subplots() 
plt.subplot(2,1,1)
plt.bar(support, [x[0] for x in omnia.values()], align = "center", color="forestgreen")
plt.semilogy()
plt.ylabel("Time (sec)")

plt.subplot(2,1,2)
plt.bar(support, [x[4] for x in omnia.values()], align = "center", color="firebrick")
plt.semilogy()
plt.ylabel("Memory (bytes)")

plt.show()
fig.tight_layout()
    