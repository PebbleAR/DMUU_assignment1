from Part1 import generate_data
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

np.random.seed(1)

n = 15          # Number of items
p_prime = 25          # Unit overflow cost
s = 4           # Salvage value
c = 15          # Unit capacity cost

## Table 2, page 476 on Capacity level
L = 30 * n
U = 120 * n

# Generate data
Data = generate_data(15)
mu = np.array(Data[0])
sigma = np.array(Data[1])
revenue = np.array(Data[2])

N = 10
M = 3

def generateD(N,mu,sigma):
    """Generate an instance of D"""
    D = np.array([norm.rvs(size = N, loc = mu[i], scale = sigma[i]) for i in range(15)])
    return D

D = generateD(N,mu,sigma)


optimalVals = []
solutionVariables = []
for j in range(M):
    D = generateD(N,mu,sigma)
    m = gp.Model("KnapsackSAA")
    x = m.addVars(n, vtype = GRB.BINARY, name="x")
    C = m.addVars(1, vtype = GRB.CONTINUOUS, name="C")
    y = m.addVars(N, vtype = GRB.CONTINUOUS, name="y")

    sum1_obj = [sum((revenue[i] - s)*D[i][j]*x[i] for i in range(15)) for j in range(N)]
    sum2_obj = sum(sum1_obj[k] - (p_prime - s) * y[k] for k in range(N))
    obj = (s-c)*C[0] + (1/N)*(sum2_obj)

    m.addConstrs((y[k]>=sum(D[i][k]*x[i] for i in range(n)) - C[0] for k in range(N)), name='c1')
    m.addConstrs((y[k]>=0 for k in range(N)), name='c2')
    m.addConstr((C[0]>=L), name='c3')
    m.addConstr((C[0]<=U), name='c4')

    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    optimalVals.append(m.objval)

    dict_vals = dict()
    for v in m.getVars():
        dict_vals[v.VarName] = v.X

    solutionVariables.append(dict_vals)
        
MaxSol = np.argmax(optimalVals)
solution = solutionVariables[MaxSol]

max(optimalVals)
solution