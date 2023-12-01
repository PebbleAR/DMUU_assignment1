import numpy as np
from itertools import product
## (SIMPLIFIED) DATA ##

# comment out one of the MADs:
# MAD = np.sqrt(2/3) 
MAD = (2/3)

m = 5 # number of actions (arcs)
n = 4 # number of states (vertices)
mu = 1
l = 0
u = 2
p = [MAD / (2*(mu-l)), 1 - MAD / (2*(mu-l)) - MAD / (2*(u-mu)), MAD / (2*(u-mu))]
nu = [mu - MAD, mu + MAD]
A = [(1,2), (1,3), (2,3), (2,4), (3,4)]
predecessor = {4: [2,3], 3:[1,2], 2: [1]}

# Determine maximal path length recursively (pseudo code is in the report)
def determineDuration3Point(alpha, i = n):
    """Recursive function to determine the critical path given weights alpha for the actions"""
    assert len(alpha) == m
    if i == 1: # no predecessor, so value is 0 
        return 0
    else:
        val = [determineDuration3Point(alpha, q) + alpha[A.index((q,i))] for q in predecessor[i]]
        return max(val)

# Upperbound
total = []
for alpha in product(range(3), repeat=5):
    prod = []
    criticalCut = determineDuration3Point(alpha)
    for a in range(5):
        prod.append(p[alpha[a]])
    total.append(np.prod(prod)*criticalCut)

print(f"Upperbound for MAD = {MAD}, is: {sum(total)}")

# Lowerbound
total = []
for alpha in product(range(2), repeat=5):
    total.append(((1/2)**5) * determineDuration3Point([nu[i] for i in alpha]))
sum(total)
print(f"Lowerbound for MAD = {MAD}, is: {sum(total)}")





    





