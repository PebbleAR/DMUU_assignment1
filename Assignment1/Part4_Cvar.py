from Part1 import generate_data
from Part2 import Monte_Carlo_1
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

np.random.seed(1)

n = 15          # Number of items
p_prime = 25    # Unit overflow cost
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

sigma_prime = sum(sigma)
mu_prime = sum(mu)

N = 500

def generateD(N,mu,sigma):
    """Generate an instance of D"""
    D = np.array([norm.rvs(size = N, loc = mu[i], scale = sigma[i]) for i in range(15)])
    return D

# Set parameters of the model
alpha = 0.0001
beta = 0.8

# Solve the linear program, using the generated D
D = generateD(N,mu,sigma)

m = gp.Model("KnapsackCVaR")
x = m.addVars(n, vtype = GRB.BINARY, name="x")
C = m.addVars(1, vtype = GRB.CONTINUOUS, name="C")
y = m.addVars(N, vtype = GRB.CONTINUOUS, name="y")
CVaR = m.addVar(1, vtype = GRB.CONTINUOUS, name="CVaR")
eta = m.addVars(N, vtype = GRB.CONTINUOUS, name="eta")
t = m.addVars(1, vtype = GRB.CONTINUOUS, name="t")
sum1_obj = [sum((revenue[i] - s)*D[i][j]*x[i] for i in range(15)) for j in range(N)]
sum2_obj = sum(sum1_obj[k] - (p_prime - s) * y[k] for k in range(N))
obj = (s-c)*C[0] + (1/N)*(sum2_obj)
new_obj = (1-beta) * obj + beta*CVaR
m.addConstrs((y[k]>=sum(D[i][k]*x[i] for i in range(n)) - C[0] for k in range(N)), name='c1')
m.addConstrs((y[k]>=0 for k in range(N)), name='c2')
m.addConstr((C[0]>=L), name='c3')
m.addConstr((C[0]<=U), name='c4')
m.addConstr((CVaR == t[0] + 1/(alpha*N)*sum(eta[k] for k in range(N))), name='c5')
m.addConstrs((eta[k] >= t[0] - ((s-c)*C[0]) + sum2_obj for k in range(N)), name='c6')
m.addConstrs((eta[k] >= 0 for k in range(N)), name='c7')
m.setObjective(obj, GRB.MAXIMIZE)
m.optimize()

dict_vals = dict()
for v in m.getVars():
    dict_vals[v.VarName] = v.X

#solution CVaR
x = list(dict_vals.values())[0:15]
C = list(dict_vals.values())[15]
output = Monte_Carlo_1(10000, x, C)
profit = output[2]
plt.hist(profit, 50)
plt.title(fr"Monte Carlo simulation of CVaR solution, with $\alpha$ = {alpha} and $\beta$ = {beta}")
plt.xlabel("Profit")
plt.show()