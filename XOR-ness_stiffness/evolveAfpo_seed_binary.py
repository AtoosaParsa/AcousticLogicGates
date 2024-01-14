from afpo_seed_binary import AFPO
import os
import constants as c
import random

#cleaning up  the data files
try:
    os.remove("savedRobotsLastGenAfpoSeed.dat")
except OSError:
    pass
try:
    os.remove("avgFitnessAfpoSeed.dat")
except OSError:
    pass
try:
    os.remove("savedRobotsAfpoSeed.dat")
except OSError:
    pass

runs = c.RUNS
for r in range(1, runs+1):
    print("*********************************************************", flush=True)
    print("run: "+str(r), flush=True)
    randomSeed = r
    random.seed(r)
    afpo = AFPO(randomSeed)
    afpo.Evolve()
    #afpo.Show_Best_Genome()
        


