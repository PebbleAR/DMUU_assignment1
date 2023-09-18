#%% Initialization
from Part1 import generate_data
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(1)

n = 15          # Number of items
p = 25          # Unit overflow cost
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


#%% Monte Carlo simulation of initial solution
C = (U + L)/2

def Monte_Carlo_1(runs, p, s, c, C):
    ## 1. Generate item sizes
    # The first seven items are in the knapsack, so we only have to generate 7 item sizes
    D = np.array([norm.rvs(size = runs, loc = Data[0][i], scale = Data[1][i]) for i in range(7)])
    
    ## 2. In cases where the simulation yields negative item sizes, these values can be adjusted to 0.
    D[np.where(D<0)] = 0 #NOTE: Does this work since D is an array of arrays?
    
    ## 3. Calculate the revenue, cost, and total profit.
    # Unit revenue
    R = np.array(Data[2][0:7]) 
    # See page 463 of the article for rewriting of total profit
    
    P_i = np.sum(D.T*(R-s), axis = 1) - (p-s)*(np.sum(D,axis = 0)-C)*(np.sum(D,axis = 0)>C) + s*C
    P = np.sum(P_i)/runs
    
    return P

## Determine Number of Runs

def number_runs(steps, epsilon):
    # Initialization
    i = 0
    P_1 = 0
    P_2 = 100
    
    # Increase number of runs until change is no longer significant
    while abs(P_1-P_2)>epsilon:
        P_1 = P_2
        i+=1
        P_2 = Monte_Carlo_1(steps*i, p, s, c, C)
    
    return i*steps

N_runs = number_runs(1000, 0.1)

P = Monte_Carlo_1(N_runs, p, s, c, C)

#%%
## 95% confidence interval for the expected profit.
# Confidence interval D
# D_bar = np.sum(D, axis = 1)/runs 
# CI_D_L = D_bar - 1.96*np.array(Data[1][0:7])/np.sqrt(runs)
# CI_D_U = D_bar + 1.96*np.array(Data[1][0:7])/np.sqrt(runs)

# Confidence interval profit
# ...?

## Construct a histogram for the profit distribution based on the simulation results.
plt.hist(P_i)
plt.show()

## Calculate the true objective function.
def loss_function_normal(C,mu,sigma):
    z = (C - mu)/sigma
    return sigma*(norm(0,1).pdf(z) - z + z*norm(0,1).cdf(z))
    
true_profit = np.sum((revenue[0:7]-s)*mu[0:7])-(p-s)*loss_function_normal(C,np.sum(mu[0:7]),np.sum(sigma[0:7])) + s*C

#%% Monte Carlo simulation of the improved solution.

#%%

data = generate_data(15) #data[0]: mu, data[1]: sigma, data[2]: unit revenue
x = np.ones(8)
y = np.zeros(8)
x = np.append(x,y)
C = U           # read page 464 for argumentation of chosen values for the constants
x[0] = (L+U)/(2*(U-L))

print(x) #x_0 is additional item
m = 100
D =  [norm.rvs(size = m, loc = data[0][i], scale = data[1][i]) for i in range(7)]  #D_i contains m itemsizes of item i+1. 

revenuelist = []
totalsizelist = []
for j in range(m):
    revenue = 0
    total_itemsize = 0
    for i in range(7):
        revenue += D[i][j] * data[2][i] 
        total_itemsize += D[i][j]
    revenuelist.append(revenue)
    totalsizelist.append(total_itemsize)

cost = [(totalsizelist[i] - (U+L)/2)*p  if (totalsizelist[i] - (U+L)/2 > 0) else (totalsizelist[i] - (U+L)/2)*s for i in range(m)]
profit = [x[0] - x[1] for x in zip(revenuelist, cost)]

print(profit)