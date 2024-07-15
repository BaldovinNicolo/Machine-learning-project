#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:35:42 2024

@author: nicolobaldovin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tracemalloc
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
import time


file = pd.read_csv("Semion/semeion.data", header=None, sep = " ")
file = file.drop(columns=file.columns[[266]])
# Link: https://archive.ics.uci.edu/dataset/178/semeion+handwritten+digit
X = file.iloc[:, :256].copy()
X = X.values
support_y = file.iloc[:, -10:].copy()
support_y["y"] = 0
max_index = support_y.values.argmax(axis=1)
support_y["y"] = max_index
y = support_y["y"].values


class Clustering:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.mixture_gaussian = {}
        self.mean_shift = {}
        self.normalized_cut = {}
        self.n = len(self.y)
    
    def rand_index(self, y_predict):
        a = 0
        b = 0 
        for j in range(self.n):
            for i in range(j+1, self.n):
                if self.y[j] == self.y[i] and y_predict[j] == y_predict[i]:
                    a = a + 1
                elif self.y[j] != self.y[i] and y_predict[j] != y_predict[i]:
                    b = b + 1
        return (2 * (a + b)) / (self.n * (self.n-1))
                    
    def classification(self):
    
        for clusters in range(5,7):                                    
            
            self.mixture_gaussian[clusters] = {}
            self.mean_shift[clusters] = {}
            self.normalized_cut[clusters] = {}
            
            for dimensions in range(2,40,28):                                           
                X_reduced = PCA(n_components=dimensions).fit_transform(self.X)
                
                self.mixture_gaussian[clusters][dimensions] = []
                self.mean_shift[clusters][dimensions] = []
                self.normalized_cut[clusters][dimensions] = []   
                

                # mixture_gaussian
                start = time.time()
                tracemalloc.start()
                
                y_predict = GaussianMixture(n_components=clusters, covariance_type="diag").fit_predict(X_reduced)
                
                rand_ind = self.rand_index(y_predict)
                non, peak = tracemalloc.get_traced_memory()
                end = time.time()
                tracemalloc.stop()
                
                self.mixture_gaussian[clusters][dimensions].append(end-start)
                self.mixture_gaussian[clusters][dimensions].append(peak)
                self.mixture_gaussian[clusters][dimensions].append(rand_ind)
                self.mixture_gaussian[clusters][dimensions].append((X_reduced, y_predict))

                
                
                # mean_shift
                start = time.time()
                tracemalloc.start()
                
                MeanS = MeanShift(bandwidth = clusters*0.1, n_jobs = 1)
                y_predict = MeanS.fit_predict(X_reduced)
                
                rand_ind = self.rand_index(y_predict)
                non, peak = tracemalloc.get_traced_memory()
                end = time.time()
                tracemalloc.stop()
                
                self.mean_shift[clusters][dimensions].append(end-start)
                self.mean_shift[clusters][dimensions].append(peak)
                self.mean_shift[clusters][dimensions].append(rand_ind)
                self.mean_shift[clusters][dimensions].append((X_reduced, y_predict))
                
                
                # normalized_cut
                start = time.time()
                tracemalloc.start()
                
                y_predict = SpectralClustering(n_clusters=clusters, affinity='nearest_neighbors').fit_predict(X_reduced)
                
                rand_ind = self.rand_index(y_predict)
                non, peak = tracemalloc.get_traced_memory()
                end = time.time()
                tracemalloc.stop()
                
                self.normalized_cut[clusters][dimensions].append(end-start)
                self.normalized_cut[clusters][dimensions].append(peak)
                self.normalized_cut[clusters][dimensions].append(rand_ind)
                self.normalized_cut[clusters][dimensions].append((X_reduced, y_predict))
                
                
                """PLOT OF CLUSTERS IN TWO DIMENSIONS"""
                if dimensions == 2:
                    fig = plt.figure(figsize=(20, 10))
                    support2D = []
                    support2D.append(self.mixture_gaussian[clusters][dimensions][3])
                    support2D.append(self.mean_shift[clusters][dimensions][3])
                    support2D.append(self.normalized_cut[clusters][dimensions][3])
                    support_title=['mixture gaussian', "mean shift", "normalized cut"]
                    for i, a in enumerate(support2D):
                        ax = fig.add_subplot(1, 3, i+1)
                        ax.scatter(a[0][:, 0], a[0][:, 1], c=a[1], cmap='rainbow')
                        ax.set_title(support_title[i])
                    # ax[0].set_title("Mixture Gaussian")
                    # ax[1].set_title("Mean Shift")
                    # ax[2].set_title("Normalized Cut")
                    plt.subplots_adjust(bottom=0.07, top=0.93, right=0.97, left=0.03, hspace=0.2, wspace=0.2)
                    plt.show()
        
    def grafico(self):
        n = ((40-2)//28)+1                                         # PLOT
        n_clusters = [i for i in range(5,7) for _ in range(n)]     
        dimension = []
        for i in range(5,7):                                       
            for j in range(2,40,28):                               
                dimension.append(j)

        mixture_gaussian_randindex = []
        mixture_gaussian_time = []
        mixture_gaussian_memory = []
        mean_shift_randindex = []
        mean_shift_time = []
        mean_shift_memory = []
        normalized_cut_randindex = []
        normalized_cut_time = []
        normalized_cut_memory = []
        for i in range(5,7):                                      
            for j in range(2,40,28):                             
                mixture_gaussian_randindex.append(self.mixture_gaussian[i][j][2])
                mixture_gaussian_time.append(self.mixture_gaussian[i][j][0])
                mixture_gaussian_memory.append(self.mixture_gaussian[i][j][1])
                
                mean_shift_randindex.append(self.mean_shift[i][j][2])
                mean_shift_time.append(self.mean_shift[i][j][0])
                mean_shift_memory.append(self.mean_shift[i][j][1])
                
                normalized_cut_randindex.append(self.normalized_cut[i][j][2])
                normalized_cut_time.append(self.normalized_cut[i][j][0])
                normalized_cut_memory.append(self.normalized_cut[i][j][1])
                
        dimension = np.array(dimension)
        n_clusters = np.array(n_clusters)
        mixture_gaussian_randindex = np.array(mixture_gaussian_randindex)        
        mixture_gaussian_time = np.array(mixture_gaussian_time)
        mixture_gaussian_memory = np.array(mixture_gaussian_memory)
        mean_shift_randindex = np.array(mean_shift_randindex)
        mean_shift_time = np.array(mean_shift_time)
        mean_shift_memory = np.array(mean_shift_memory)
        normalized_cut_randindex = np.array(normalized_cut_randindex)
        normalized_cut_time = np.array(normalized_cut_time)
        normalized_cut_memory = np.array(normalized_cut_memory)
        
        '''PLOT OF ACCURACY, TIME AND MEMORY IN THREE DIMENSIONS'''
        fig = plt.figure(figsize=(20, 10))      
        ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=2, projection='3d')
        ax1.scatter(n_clusters, dimension, mixture_gaussian_randindex, marker="o", c=n_clusters, cmap='viridis', s=100)
        for valore_x in np.unique(n_clusters):
            x_ind = n_clusters == valore_x 
            x_data = n_clusters[x_ind]
            y_data = dimension[x_ind]
            z_data = mixture_gaussian_randindex[x_ind]
            ax1.set_xlabel("Clusters")
            ax1.set_ylabel("Dimension")
            ax1.set_zlabel("Rand index")
            ax1.plot(x_data, y_data, z_data, linewidth = 1, c = "black")
        plt.draw()
        ax1.view_init(10,100)
        plt.show()
        fig = plt.figure(figsize=(20, 10))  
        ax2 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=2, projection='3d')
        ax2.scatter(n_clusters, dimension, mixture_gaussian_time, marker="o", c=n_clusters, cmap='viridis', s=100)
        for valore_x in np.unique(n_clusters):
            x_ind = n_clusters == valore_x 
            x_data = n_clusters[x_ind]
            y_data = dimension[x_ind]
            z_data = mixture_gaussian_time[x_ind]
            ax2.set_xlabel("Clusters")
            ax2.set_ylabel("Dimension")
            ax2.set_zlabel("Time")
            ax2.plot(x_data, y_data, z_data, linewidth = 1, c = "black")
        ax2.view_init(10,100)
        plt.show()
        fig = plt.figure(figsize=(20, 10))  
        ax3 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=2, projection='3d')
        ax3.scatter(n_clusters, dimension, mixture_gaussian_memory, marker="o", c=n_clusters, cmap='viridis', s=100)
        for valore_x in np.unique(n_clusters):
            x_ind = n_clusters == valore_x 
            x_data = n_clusters[x_ind]
            y_data = dimension[x_ind]
            z_data = mixture_gaussian_memory[x_ind]
            ax3.set_xlabel("Clusters")
            ax3.set_ylabel("Dimension")
            ax3.set_zlabel("Memory")
            ax3.plot(x_data, y_data, z_data, linewidth = 1, c = "black")
        plt.show()
        
        fig = plt.figure(figsize=(20, 10))      
        ax4 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=2, projection='3d')
        ax4.scatter(n_clusters, dimension, mean_shift_randindex, marker="o", c=n_clusters, cmap='viridis', s=100)
        for valore_x in np.unique(n_clusters):
            x_ind = n_clusters == valore_x 
            x_data = n_clusters[x_ind]
            y_data = dimension[x_ind]
            z_data = mean_shift_randindex[x_ind]
            ax4.set_xlabel("Kernel Width")
            ax4.set_ylabel("Dimension")
            ax4.set_zlabel("Rand index")
            ax4.plot(x_data, y_data, z_data, linewidth = 1, c = "black")
        plt.draw()
        ax4.view_init(10,100)
        plt.show()
        fig = plt.figure(figsize=(20, 10))  
        ax5 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=2, projection='3d')
        ax5.scatter(n_clusters, dimension, mean_shift_time, marker="o", c=n_clusters, cmap='viridis', s=100)
        for valore_x in np.unique(n_clusters):
            x_ind = n_clusters == valore_x 
            x_data = n_clusters[x_ind]
            y_data = dimension[x_ind]
            z_data = mean_shift_time[x_ind]
            ax5.set_xlabel("Kernel Width")
            ax5.set_ylabel("Dimension")
            ax5.set_zlabel("Time")
            ax5.plot(x_data, y_data, z_data, linewidth = 1, c = "black")
        ax5.view_init(10, 100)
        plt.show()
        fig = plt.figure(figsize=(20, 10))  
        ax6 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=2, projection='3d')
        ax6.scatter(n_clusters, dimension, mean_shift_memory, marker="o", c=n_clusters, cmap='viridis', s=100)
        for valore_x in np.unique(n_clusters):
            x_ind = n_clusters == valore_x 
            x_data = n_clusters[x_ind]
            y_data = dimension[x_ind]
            z_data = mean_shift_memory[x_ind]
            ax6.set_xlabel("Kernel Width")
            ax6.set_ylabel("Dimension")
            ax6.set_zlabel("Memory")
            ax6.plot(x_data, y_data, z_data, linewidth = 1, c = "black")
        plt.show()    
        
        fig = plt.figure(figsize=(20, 10))      
        ax7 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=2, projection='3d')
        ax7.scatter(n_clusters, dimension, normalized_cut_randindex, marker="o", c=n_clusters, cmap='viridis', s=100)
        for valore_x in np.unique(n_clusters):
            x_ind = n_clusters == valore_x 
            x_data = n_clusters[x_ind]
            y_data = dimension[x_ind]
            z_data = normalized_cut_randindex[x_ind]
            ax7.set_xlabel("Clusters")
            ax7.set_ylabel("Dimension")
            ax7.set_zlabel("Rand index")
            ax7.plot(x_data, y_data, z_data, linewidth = 1, c = "black")
        plt.draw()
        ax7.view_init(10,100)
        plt.show()
        fig = plt.figure(figsize=(20, 10))  
        ax8 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=2, projection='3d')
        ax8.scatter(n_clusters, dimension, normalized_cut_time, marker="o", c=n_clusters, cmap='viridis', s=100)
        for valore_x in np.unique(n_clusters):
            x_ind = n_clusters == valore_x 
            x_data = n_clusters[x_ind]
            y_data = dimension[x_ind]
            z_data = normalized_cut_time[x_ind]
            ax8.set_xlabel("Clusters")
            ax8.set_ylabel("Dimension")
            ax8.set_zlabel("Time")
            ax8.plot(x_data, y_data, z_data, linewidth = 1, c = "black")
        ax8.view_init(10,100)
        plt.show()
        fig = plt.figure(figsize=(20, 10))  
        ax9 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), rowspan=2, projection='3d')
        ax9.scatter(n_clusters, dimension, normalized_cut_memory, marker="o", c=n_clusters, cmap='viridis', s=100)
        for valore_x in np.unique(n_clusters):
            x_ind = n_clusters == valore_x 
            x_data = n_clusters[x_ind]
            y_data = dimension[x_ind]
            z_data = normalized_cut_memory[x_ind]
            ax9.set_xlabel("Clusters")
            ax9.set_ylabel("Dimension")
            ax9.set_zlabel("Memory")
            ax9.plot(x_data, y_data, z_data, linewidth = 1, c = "black")
        plt.show()  
                
        
if __name__ == '__main__':
    result = Clustering(X,y)
    result.classification()      
    result.grafico()


            