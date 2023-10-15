from Part1 import generate_data
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import time
from Part2 import Monte_Carlo_1, improved_solution, confidence_interval

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

N = 1000
M = 5

def generateD(N,mu,sigma):
    """Generate an instance of D"""
    D = np.array([norm.rvs(size = N, loc = mu[i], scale = sigma[i]) for i in range(15)])
    return D

start_time = time.time()

### SAA
optimalVals = []
solutionVariables = []
for j in range(M):  
    # Solve the linear program, using the generated D 
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
optimalVal = max(optimalVals)
optimalVal
list(solution.values())[0:15] 
list(solution.values())[15] 

### Determining the gap 
## Determine Lowerbound
t = 2.132       # alpha = 0.05, M = 5
# t = 1.833       # alpha = 0.05, M = 10
# t = 1.729       # alpha = 0.05, M = 20

L = np.mean(optimalVals) - t * np.var(optimalVals)
np.mean(optimalVals)
np.var(optimalVals)

## Determine Upperbound
N_prime = 100000 # Must be much larger than N

x = list(solution.values())[0:15]   # SAA solution
C = list(solution.values())[15]     # SAA solution

D = generateD(N_prime,mu,sigma)     

y = [max(sum(D[i][k]*x[i] for i in range(n)) - C, 0) for k in range(N_prime)]

sum1_obj = [sum((revenue[i] - s)*D[i][j]*x[i] for i in range(15)) for j in range(N_prime)]
sum2_obj = [sum1_obj[k] - (p_prime - s) * y[k] for k in range(N_prime)]

std = np.std(sum2_obj)
mean  = (s-c)*C + (1/N_prime)*(sum(sum2_obj))

z = 1.64
U = mean + z*std
gap = U - L

print(L, optimalVal, U, gap)

print("--- %s seconds ---" % (time.time() - start_time))

# Confidence interval for the difference of the mean profits
def CI_diff_mean_profits(runs, data):
    diff_profits = []
    for _ in range(runs):
        sol_SAA = Monte_Carlo_1(3225, x, C)
        mean_prof_SAA = np.mean(sol_SAA[2])

        sol_improved = improved_solution(data, sol_SAA[3])
        mean_prof_I = np.mean(sol_improved[2])

        diff_profits.append(abs(mean_prof_SAA - mean_prof_I))

    return confidence_interval(diff_profits), sol_SAA[2]

CI_diff_mean_prof, sol_SAAt = CI_diff_mean_profits(100, Data)
print(f"The confidence interval of the difference of the mean profits with 100 runs is: {CI_diff_mean_prof}")

# Histogram of the SAA solution
plt.hist(sol_SAAt, 50)
plt.title("Histogram of Monte Carlo simulation of SAA solution")
plt.xlabel("Profit")
plt.show()