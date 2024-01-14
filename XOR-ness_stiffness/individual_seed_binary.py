from switch_binary import switch
import constants as c
import random
import math
import numpy as np
import sys
import pickle

class INDIVIDUAL:
    def __init__(self, i):
        self.m1 = 1
        self.m2 = 10 #[2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000] #10 is what is in fig2 of the paper
        self.N_light = 9 #[25, 50, 75] #[9, 15, 21]
        self.N = 30
        #self.low = 7
        #self.high = 12
        #indices = random.sample(range(0, N), N_light)
        self.genome = np.random.randint(low=0, high=2, size=self.N)#random.sample(range(0, self.N), self.N_light) #np.random.randint(0, high=c.GRID_SIZE-1, size=(c.FIBERS*2, 4), dtype='int')
        self.fitness = 0
        self.ID = i
    
    def Compute_Fitness(self, show=False):
        # wait for the simulation to end and get the fitness
        self.fitness = switch.evaluate(self.m1, self.m2, self.N_light, self.genome)#, self.low, self.high)
        if show:
            switch.showPacking(self.m1, self.m2, self.N_light, self.genome)#, self.low, self.high)
            print("fitness is:")
            print(self.fitness)
        return self.fitness
    
    def Mutate(self):
        mutationRate = 0.05
        probToMutate = np.random.choice([False, True], size=self.genome.shape, p=[1-mutationRate, mutationRate])
        candidate = np.where(probToMutate, 1-self.genome, self.genome)
        
        self.genome = candidate
        
    
    def Print(self):
        print('[', self.ID, self.fitness, ']', end=' ')
        
    
    def Save(self):
        f = open('savedFitnessSeed.dat', 'ab')
        pickle.dump(self.fitness , f)
        f.close()
    
    def SaveBest(self):
        f = open('savedBestsSeed.dat', 'ab')
        pickle.dump(self.genome , f)
        f.close()