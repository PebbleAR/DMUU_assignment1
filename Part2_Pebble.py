#%% Initialization
from Part1 import generate_data
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(2)

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

def Monte_Carlo_1(runs, p, s, C):
    ## 1. Generate item sizes
    # The first seven items are in the knapsack, so we only have to generate 7 item sizes
    D = np.array([norm.rvs(size = runs, loc = Data[0][i], scale = Data[1][i]) for i in range(7)])
    
    ## 2. In cases where the simulation yields negative item sizes, these values can be adjusted to 0.
    D[np.where(D<0)] = 0 
    
    ## 3. Calculate the revenue, cost, and total profit.
    # Unit revenue
    R = np.array(Data[2][0:7]) 
    # See page 463 of the article for rewriting of total profit
    
    P_i = np.sum(D.T*(R-s), axis = 1) - (p-s)*(np.sum(D,axis = 0)-C)*(np.sum(D,axis = 0)>C) + s*C
    P = np.sum(P_i)/runs
    
    return P

def Monte_Carlo_pebble(runs, p, s, C):
    ## 1. Generate item sizes
    # The first seven items are in the knapsack, so we only have to generate 7 item sizes
    D = np.array([norm.rvs(size = 3, loc = Data[0][i], scale = Data[1][i]) for i in range(7)])

    ## 2. In cases where the simulation yields negative item sizes, these values can be adjusted to 0.
    D[np.where(D<0)] = 0

    ## 3. Calculate the revenue, cost, and total profit.
    R = np.array(Data[2][0:7]) # unit revenue
    revenue = np.matmul(D.T, R)  # total revenue (revenue[i] is the revenue of one run)
    total_size = np.sum(D, axis=0) # The item sizes of items together (total_size[i] is the total size of one run)
    remaining_capacity = C - total_size
    cost = np.where(remaining_capacity > 0, -s*remaining_capacity, p*remaining_capacity) # if the remaining capcity is positive, we use the salvage value otherwise the cost value p
    profit = revenue - cost
    return revenue, cost, profit

def confidence_interval(profit):
    mean = np.mean(profit)
    std = np.std(profit)
    Z_ALPHA = 1.960
    return [mean - Z_ALPHA*std/np.sqrt(len(profit)), mean + Z_ALPHA*std/np.sqrt(len(profit)) ]


revenue, cost, profit = Monte_Carlo_pebble(320, p, s, C)
CI = confidence_interval(profit)
np.mean(profit)

CI[1]-CI[0]


plt.hist(profit)
plt.show()




# D = np.array([norm.rvs(size = 3, loc = Data[0][i], scale = Data[1][i]) for i in range(3)])
# D[np.where(D<0)] = 0 
# R = np.array(Data[2][0:3])
# R
# D
# total_size = np.sum(D, axis=0)
# total_size

# total_size - C

# C - total_size


# np.matmul(D.T, R)

