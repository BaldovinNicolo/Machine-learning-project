#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:25:17 2023

@author: nicolobaldovin
"""

import numpy as np
import time
import tracemalloc
import copy
import math
import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statistics import mean

# FIRST METHOD (CONSTRAINT-PROPAGATION-BACKTRACKING)

class Scacchiera:   
    def __init__(self, n):
        n = int(n)
        self.n = n
        self.board = np.array([["." for j in range(n)] for k in range(n)])      
        self.queue = []
        
    def stampa(self):
        for f in range(self.n):
            for k in range(self.n):
                print(self.board[f][k], end = " ")
            print()
                
    def copia(self):
        chessboard = Scacchiera()
        chessboard.board = self.board.copy()
        return chessboard
    
    def add(self, queens):
        for i in queens:
            self.queue.append((i[0],i[1]))
            
    def conditions(self,row,column):
        
        row_sx = [_ for _ in range(row,-1,-1)]
        col_down = [_ for _ in range(column, self.n)]
        row_dx = [_ for _ in range(row, self.n)]
        col_up = [_ for _ in range(column, -1, -1)]
        top_right = zip(row_sx,col_down)
        bottom_right = zip(row_dx,col_down)
        bottom_left = zip(row_dx,col_up)
        top_left = zip(row_sx,col_up)

        
        for l in range(self.n):
            if self.board[l][column] == "Q":
                return False
            
        for m in range(self.n):
            if self.board[row][m] == "Q":
                return False  
    
        for k in top_right:   
            if self.board[k[0]][k[1]] == "Q":
                return False
            
        for k in bottom_right:  
            if self.board[k[0]][k[1]] == "Q":
                return False

        for k in bottom_left:  
            if self.board[k[0]][k[1]] == "Q":
                return False
            
        for k in top_left:   
            if self.board[k[0]][k[1]] == "Q":
                return False           
            
        return True
    
    
    def solve(self):
        self.add([(0,0)])
        k = self.queue.pop()
        k = list(k)
        while True:
            while k[0] < self.n: 
                if self.conditions(k[0], k[1]):
                    self.queue.append((k[0],k[1]))
                    self.board[k[0]][k[1]] = "Q"
                    k[0] = 0
                    break
                k[0] = k[0] + 1           
            else:    
                if self.queue == False:
                    return False
                k = self.queue.pop()    
                k = list(k)
                self.board[k[0]][k[1]] = "."
                k[0] = k[0] + 1         
                continue
            if k[1] == int(self.n - 1):
                return True
            k[1] = k[1] + 1
            

def Solution(n):
    if n < 4:
        return False
    s = Scacchiera(n)
    if s.solve():
        return True, s.board
    else:
        return False

  

# SECOND METHOD (SIMULATED-ANNEALING)    

def obj_funz(chessboard):
    obj_val = 0
                
    dict_individual = {}
    for i in chessboard:
        if i not in dict_individual.keys():
            dict_individual[i] = 0
        else:
            dict_individual[i] = 1 + dict_individual[i]
        for (k,v) in dict_individual.items():
            if v != 0:
                obj_val = obj_val + v
    
    for x in range(len(chessboard)):                         
        for y in range(x+1,len(chessboard)):   
            if x != y:
                delta_x = abs(x-y)
                delta_y = abs(chessboard[x]-chessboard[y])
                if delta_x == delta_y :                     
                    obj_val = obj_val + 1                   
    return obj_val
 

def stampa(N, chessboard):
    layout=[]
    en_chessboard = []
    for k in range(N):
        element = chessboard[k]
        en_chessboard.append(element)
    layout = np.full((N,N), 0)
    k = 0
    for j in en_chessboard:
        layout[j][k] = 1
        k = k + 1                                       
        
def creation_chessboard(N):
    chessboard = []
    for i in range(N):
        n = random.randint(0,N-1)
        chessboard.append(n)
    return chessboard


def solution(N,temp_0= 1.0,cooling_rate = 0.95):
    if N < 4:
        return False
    chessboard_0 = creation_chessboard(N)
    current_obj_val = obj_funz(chessboard_0)     
    best_obj_val = current_obj_val
    temperature = temp_0
    best_chessboard = chessboard_0.copy()
    while True:
        current_chessboard = best_chessboard.copy()
        x = random.randint(0, len(chessboard_0)-1)       
        y = random.randint(0, len(chessboard_0)-1) 
        current_chessboard[x] = y             
        new_obj_val = obj_funz(current_chessboard) 
        prob=random.random()
        
        if new_obj_val < best_obj_val:
            best_obj_val=new_obj_val                         
            best_chessboard=current_chessboard.copy()
            temperature = cooling_rate * temperature
            
        elif prob<math.exp(-(new_obj_val-best_obj_val) / temperature):
            best_obj_val=new_obj_val                         
            best_chessboard=current_chessboard.copy()
        
        if new_obj_val == 0:           
            best_chessboard=current_chessboard.copy() 
            break
 
    return best_chessboard


# THIRD METHOD (GENETIC-ALGORITHMS)

class Genotype:
    def __init__(self,n,genotype = None):
        self.n = n
        self.genotype = genotype
        if not self.genotype: 
            self.count_scontri = 0
            self.creation()
        self.count_scontri = self.fitness()
        
    def __iter__(self):
        return self
        
    def creation(self):
        self.genotype = []
        while len(self.genotype) < self.n:      
            i = random.randint(0,self.n-1)
            if i not in self.genotype:
                self.genotype.append(i)
            
    
    def stampa(self):
        layout=[]
        en_chessboard = []
        for k in range(self.n):
            element = self.genotype[k]
            en_chessboard.append(element)
        layout = np.full((self.n,self.n), 0)
        k = 0
        for j in en_chessboard:
            layout[j][k] = 1
            k = k + 1  
        
    def fitness(self):
        count_scontri = 0
        for i in range(len(self.genotype)):
            for j in range(i+1, len(self.genotype)):
                if i != j:
                   if abs(i-j) == abs(self.genotype[i] - self.genotype[j]):
                       count_scontri = 1 + count_scontri
        self.count_scontri = count_scontri
        return self.count_scontri
    
    def mutate(self):
        prob = random.random()
        if prob > 0.90:
            a = random.randint(0,self.n-1)
            b = random.randint(0,self.n-1)
            if b in self.genotype:
                while b not in self.genotype:
                    b = random.randint(0,self.n-1)
                    self.genotype[a] = b
            else:
                self.genotype[a] = b
        self.fitness()
    
    
class Genetic:
    def __init__(self, n, pool = None):
        self.n = n
        self.pool = pool
        if not self.pool:
            self.pool = []
        
    def popolation(self):                       
        self.pool = []
        for i in range(self.n*40):
            element = Genotype(self.n)
            element.creation()
            self.pool.append(element)
    
    def mating_pool(self):
        self.fitness_dict = {}
        for j in range(len(self.pool)):
            element = self.pool[j].fitness()
            self.fitness_dict[self.pool[j]] = element
        inv_list = [(g,f) for (f,g) in self.fitness_dict.items()]
        somma_fitness=0
        for el in inv_list:
            somma_fitness+=el[0]
        probabilities=[]
        for el in inv_list:
            probabilities.append(1-(el[0]/somma_fitness))
        l = random.choices(self.pool,probabilities,k=self.n)  
        return Genetic(self.n,l)
    
    def cross_over(self, a, b):                 
        prob=random.random()
        if prob>0.1:
            k = random.randint(0,self.n)
            figlio = a.genotype[:k]
            figlia = b.genotype[:k]
            for i in range(k,self.n):
                for dati in b.genotype:
                    if dati not in figlio:
                        figlio.append(dati)
                for dati in a.genotype:
                    if dati not in figlia:
                        figlia.append(dati)
            a.genotype = figlio
            b.genotype = figlia
            return (Genotype(self.n,a.genotype), Genotype(self.n,b.genotype))
        else:
            return (Genotype(self.n,a.genotype), Genotype(self.n,b.genotype))
    
    def add(self,el):
        self.pool.append(el)
        
    def riordino(self):
        self.fitness_dict = {}
        for j in range(len(self.pool)):
            element = self.pool[j].fitness()
            self.fitness_dict[self.pool[j]] = element
        inv_list = [(g,f) for (f,g) in self.fitness_dict.items()]
        sorted_list = sorted(inv_list, key = lambda x : x[0])
        return Genetic(int(len(self.pool)), [f for (g,f) in sorted_list[:int(len(self.pool))]])  
    
def run(N,old_population):
    while True:
        bests=old_population.mating_pool()
        for k in bests.pool:
            if k.count_scontri==0:
                return k
            
        t=random.randint(0,int(N)-1)
        r=random.randint(0,int(N)-1)
        padre=Genotype(N,bests.pool[t].genotype)
        madre=Genotype(N,bests.pool[r].genotype)  
        while madre.genotype == padre.genotype:
            r = random.randint(0,int(N)-1)
            madre = Genotype(N,bests.pool[r].genotype)
        children=bests.cross_over(madre, padre)
        if children[0].count_scontri==0:
            return children[0]
        if children[1].count_scontri==0:
            return children[1]     
        children[0].mutate()
        children[1].mutate()
        old_population.add(children[0])
        old_population.add(children[1])
        for i in old_population.pool:
            if i.genotype == madre.genotype:
                old_population.pool.remove(i)
                break
        for i in old_population.pool:
            if i.genotype == padre.genotype:
                old_population.pool.remove(i)
                break
        if children[0].count_scontri==0:
            return children[0]
        if children[1].count_scontri==0:
            return children[1] 


def soluzione(N):
    if N < 4:
        return False
    else:
        initial_state = Genetic(N)
        initial_state.popolation()
        run(N,initial_state)



if __name__ == '__main__':
    
    timesCS = []
    memoryCS = []
    timesSA = []
    memorySA = []
    timesGA = []
    memoryGA = []
    
    times_CS = {}
    memory_CS = {}
    times_SA = {}
    memory_SA = {}
    times_GA = {}
    memory_GA = {}
    
    ap_times_CS = {}
    ap_memory_CS = {}
    ap_times_SA = {}
    ap_memory_SA = {}
    ap_times_GA = {}
    ap_memory_GA = {}
     
    for f in range(4,14):
        
        times_CS[f] = []
        memory_CS[f] = []
        times_SA[f] = []
        memory_SA[f] = [] 
        times_GA[f] = []
        memory_GA[f] = [] 
        
        ap_times_CS[f] = []
        ap_memory_CS[f] = []
        ap_times_SA[f] = []
        ap_memory_SA[f] = [] 
        ap_times_GA[f] = []
        ap_memory_GA[f] = [] 
        
    for k in range(5):
        
        for f in range(4,14):
            
            start = time.time()
            tracemalloc.start()
            Solution(f) # CS
            non, peak = tracemalloc.get_traced_memory()
            end = time.time()
            tracemalloc.stop()
            times_CS[f].append(end-start)
            ap_times_CS[f].append(end-start)
            memory_CS[f].append(peak)
            ap_memory_CS[f].append(peak)
        
            start = time.time()
            tracemalloc.start()
            solution(f) # SA
            non, peak = tracemalloc.get_traced_memory()
            end = time.time()
            tracemalloc.stop()
            times_SA[f].append(end-start)
            ap_times_SA[f].append(end-start)
            memory_SA[f].append(peak)
            ap_memory_SA[f].append(peak)

            start = time.time()
            tracemalloc.start()
            soluzione(f) # GA
            non, peak = tracemalloc.get_traced_memory()
            end = time.time()
            tracemalloc.stop()
            times_GA[f].append(end-start)
            memory_GA[f].append(peak)
            ap_times_GA[f].append(end-start)
            ap_memory_GA[f].append(peak)

            
    for f in range(4,14):
        
        timesCS.append(mean(times_CS[f]))
        memoryCS.append(mean(memory_CS[f]))
        timesSA.append(mean(times_SA[f]))
        memorySA.append(mean(memory_SA[f]))
        timesGA.append(mean(times_GA[f]))
        memoryGA.append(mean(memory_GA[f]))
        
    fig, axes = plt.subplots() 
    plt.subplot(2,1,1)
    plt.plot(range(4,14),timesCS, label=' CS', linewidth=2, color="purple")
    plt.plot(range(4,14),timesSA, label=' SA', linewidth=2, color = "green")
    plt.plot(range(4,14),timesGA, label=' GA', linewidth=2, color = "blue")
    plt.semilogy()
    legend = plt.legend(["CS", "SA", "GA"], loc=2, fontsize = 8)
    plt.ylabel("Times (sec)")

    plt.subplot(2,1,2)
    plt.plot(range(4,14), memoryCS, label=' CS', linewidth=2, color = "purple")
    plt.plot(range(4,14), memorySA, label=' SA', linewidth=2, color = "green")
    plt.plot(range(4,14), memoryGA, label=' GA', linewidth=2, color = "blue")
    legend = plt.legend(["CS", "SA", "GA"], loc=2, fontsize = 8)
    plt.semilogy()
    plt.ylabel("Memory (bytes)")

    plt.show()
    fig.tight_layout()

     
    