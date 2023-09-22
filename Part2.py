### PART 2.1

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
    D = np.array([norm.rvs(size = runs, loc = Data[0][i], scale = Data[1][i]) for i in range(7)])

    ## 2. In cases where the simulation yields negative item sizes, these values can be adjusted to 0.
    D[np.where(D<0)] = 0

    ## 3. Calculate the revenue, cost, and total profit.
    R = np.array(Data[2][0:7]) # unit revenue
    revenue = np.matmul(D.T, R)  # total revenue (revenue[i] is the revenue of one run)
    total_size = np.sum(D, axis=0) # The item sizes of items together (total_size[i] is the total size of one run)
    remaining_capacity = C - total_size
    cost = np.where(remaining_capacity > 0, -s*remaining_capacity, p*remaining_capacity) # if the remaining capcity is positive, we use the salvage value otherwise the cost value p
    profit = revenue - cost                                                              # NOTE: add cost for capacity
    return revenue, cost, profit

def confidence_interval(profit):
    mean = np.mean(profit)
    std = np.std(profit)
    Z_ALPHA = 1.960
    return [mean - Z_ALPHA*std/np.sqrt(len(profit)), mean + Z_ALPHA*std/np.sqrt(len(profit)) ]

### Determining CI 
# Note that in order to half the confidence interval, one needs to use 4*n runs instead of n. 
rev, cost, profit = Monte_Carlo_pebble(5, p, s, C) 
CI = confidence_interval(profit)
CI[1]-CI[0] # gap is about 30, so we need to half the interval at least 5 times to get a gap of 1. This means that we need 5120 runs. 

rev, cost, profit = Monte_Carlo_pebble(5120, p, s, C)
CI = confidence_interval(profit)
CI[1]-CI[0] # indeed the gap is smaller than 1. 
np.mean(profit)
### plot histogram
plt.hist(profit, 50)
plt.show()



### standard normal loss function

## Calculate the true objective function.
def loss_function_normal(C,mu,sigma):
    z = (C - mu)/sigma
    return sigma*(norm(0,1).pdf(z) - z + z*norm(0,1).cdf(z))
    
true_profit = np.sum((revenue[0:7]-s)*mu[0:7])-(p-s)*loss_function_normal(C,np.sum(mu[0:7]),np.sum(sigma[0:7])) + s*C
true_profit
### Part 2.2
### greedy algorithm: if the cost is negative in 10 instances, add the next item.  
def Monte_Carlo_2_pebble(runs, p, s, C):
    L = 30 * 15
    U = 120 * 15
    nr_items = 0
    cost_negative = 1
    while cost_negative:
        nr_items +=1
        ## 1. Generate item sizes
        # The first seven items are in the knapsack, so we only have to generate 7 item sizes
        D = np.array([norm.rvs(size = 10, loc = Data[0][i], scale = Data[1][i]) for i in range(nr_items)])

        ## 2. In cases where the simulation yields negative item sizes, these values can be adjusted to 0.
        D[np.where(D<0)] = 0

        ## 3. Calculate the cost 
        total_size = np.sum(D, axis=0) # The item sizes of items together (total_size[i] is the total size of one run)
        remaining_capacity = C - total_size
        cost = np.where(remaining_capacity > 0, -s*remaining_capacity, p*remaining_capacity) # if the remaining capcity is positive, we use the salvage value otherwise the cost value p
        positive_costs = np.where(cost > 0, 1, 0)
        
        if sum(positive_costs) >= 1 or nr_items == 15:
            cost_negative = 0
            

    nr_items -= 1 #we select the number of items, where the cost were not negative. 
    # print("Number of items:", nr_items)

    D = np.array([norm.rvs(size = runs, loc = Data[0][i], scale = Data[1][i]) for i in range(nr_items)])

    ## 2. In cases where the simulation yields negative item sizes, these values can be adjusted to 0.
    D[np.where(D<0)] = 0

    ## 3. Calculate the revenue, cost, and total profit.
    R = np.array(Data[2][0:nr_items]) # unit revenue
    revenue = np.matmul(D.T, R)  # total revenue (revenue[i] is the revenue of one run)
    total_size = np.sum(D, axis=0) # The item sizes of items together (total_size[i] is the total size of one run)
    remaining_capacity = C - total_size
    cost = np.where(remaining_capacity > 0, -s*remaining_capacity, p*remaining_capacity) # if the remaining capcity is positive, we use the salvage value otherwise the cost value p
    profit = revenue - cost
    return revenue, cost, profit

rev, cost, profit = Monte_Carlo_2_pebble(10240, p, s, C)
CI = confidence_interval(profit)
CI[1]-CI[0] # indeed the gap is smaller than 1. 
np.mean(profit)
### plot histogram
plt.hist(profit, 50)
plt.show()

## Compare the 'initial solution' with the 'improved solution'

def CI_diff_mean_profits(runs):
    diff_profits = []
    for _ in range(runs):
        sol_initial = Monte_Carlo_pebble(10240, p, s, C)
        mean_prof_in = np.mean(sol_initial[2])

        sol_improved = Monte_Carlo_2_pebble(10240, p, s, C)
        mean_prof_imp = np.mean(sol_improved[2])

        diff_profits.append(mean_prof_imp - mean_prof_in)

    return confidence_interval(diff_profits)

CI_diff_mean_prof = CI_diff_mean_profits(100)
print(CI_diff_mean_prof)