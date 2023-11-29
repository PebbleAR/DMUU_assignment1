import numpy as np
from itertools import product
MAD = 2/3 # equal to variance
m = 5 # number of actions (arcs)
n = 4 # number of states (vertices)
mu = 1
l = 0
u = 2
p = [MAD / (2*(mu-l)), 1 - MAD / (2*(mu-l)) - MAD / (2*(u-mu)), MAD / (2*(u-mu))]
tau = [l, mu, u]
V = [1,2,3,4]
A = [(1,2), (1,3), (2,3), (2,4), (3,4)]

predecessor = {4: [2,3], 3:[1,2], 2: [1]}

def determineDuration3Point(alpha, i = n):
    """Recursive function to determine the critical path given weights alpha for the actions"""
    assert len(alpha) == m
    if i == 1: # no predecessor, so value is 0 
        return 0
    else:
        val = [determineDuration3Point(alpha, q) + alpha[A.index((q,i))] for q in predecessor[i]]
        return max(val)

# simplified verion of lemma 2 (for the chosen values it reduces back to this)   
total = []
for alpha in product(range(3), repeat=5):
    total.append((1/3)**5 * determineDuration3Point(alpha)**5)

sum(total)

# (not the simplified version)
# This should determine the supremum given in lemma 2 of the article (for the instance of example 2)
total = []
for alpha in product(range(3), repeat=5):
    prod = []
    criticalCut = determineDuration3Point(alpha)
    for a in range(5):
        prod.append(p[alpha[a]] * criticalCut)
    ### consider each alpha with same critical cut:
    ## To be implemented
    total.append(np.prod(prod))

p


sum(total) / len(total)


    





