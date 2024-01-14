from switch_binary import switch
import matplotlib.pyplot as plt
import random
import numpy as np

m1 = 1
m2 = 10

N = 30
N_light = 9

outs = []
for i in range(0, 1001):
    print(i)
    sample = np.random.randint(low=0, high=2, size=N)
    outs.append(switch.evaluate(m1, m2, N_light, sample))

print(outs)
n, bins, patches = plt.hist(x=outs, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)#, grid=True)
plt.xlabel('Andness')
plt.ylabel('Counts')
plt.title('Random Search')
#plt.xlim([0, 8])
plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
plt.show()
plt.savefig("histogram.jpg", dpi=300)