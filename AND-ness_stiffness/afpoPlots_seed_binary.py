import pickle
import matplotlib.pyplot as plt
from switch_binary import switch
import constants as c
import numpy

runs = c.RUNS
gens = c.numGenerations
fitnesses = numpy.zeros([runs, gens])
temp = []
individuals = []
with open('savedRobotsAfpoSeed.dat', "rb") as f:
    for r in range(1, runs+1):
        for g in range(1, gens+1):
            try:
                temp.append(pickle.load(f).fitness)
            except EOFError:
                break
        fitnesses[r-1] = temp
        temp = []
f.close()

mean_f = numpy.mean(fitnesses, axis=0)
std_f = numpy.std(fitnesses, axis=0)

plt.figure(figsize=(6.4,4.8))
plt.plot(list(range(1, gens+1)), mean_f, color='blue', label="Average")
plt.fill_between(list(range(1, gens+1)), mean_f-std_f, mean_f+std_f, color='cornflowerblue', alpha=0.2, label="Standard Deviation")
plt.xlabel("Generations", fontsize=18)
plt.ylabel("Fitness (\"AND-ness\"; Eqn. 2)", fontsize=18)
plt.title("Evolutionary Search", fontsize=18)
plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
plt.tight_layout()
plt.legend(loc='upper left', )
#plt.savefig("compare.pdf")
plt.show()


# running the best individuals
m1 = 1
m2 = 10
N_light = 9
N = 30

bests = numpy.zeros([runs, gens])
temp = []
rubish = []
with open('savedRobotsLastGenAfpoSeed.dat', "rb") as f:
    for r in range(1, runs+1):
        # population of the last generation
        temp = pickle.load(f)
        # best individual of last generation
        best = temp[0]
        switch.showPacking(m1, m2, N_light, best.indv.genome)
        print(switch.evaluate(m1, m2, N_light, best.indv.genome))
        print(switch.evaluateAndPlot(m1, m2, N_light, best.indv.genome))
        temp = []
f.close()

# running all of the individuals of the last generation of each of the runs
bests = numpy.zeros([runs, gens])
temp = []
rubish = []
with open('savedRobotsLastGenAfpoSeed.dat', "rb") as f:
    for r in range(1, runs+1):
        print("run:")
        print(r)
        # population of the last generation
        temp = pickle.load(f)
        for g in range(0, gens):
            switch.showPacking(m1, m2, N_light, temp[g].indv.genome)
            print(switch.evaluateAndPlot(m1, m2, N_light, temp[g].indv.genome))
        temp = []
f.close()