### PART 2.1

#%% Initialization
from Part1 import generate_data
import numpy as np
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt


np.random.seed(1)

n       = 15          # Number of items
p_prime = 25          # Unit overflow cost
s       = 4           # Salvage value
c       = 15          # Unit capacity cost
p       = p_prime - s

initial_x = np.append(np.ones(7),np.zeros(8))

## Table 2, page 476 on Capacity level
L = 30 * n
U = 120 * n

# Generate data
Data = generate_data(15)
mu = np.array(Data[0])
sigma = np.array(Data[1])
unit_revenue = np.array(Data[2])

N = np.sum(initial_x).astype(int)



#%% Part 2.1 - Monte Carlo simulation of initial solution
C = (U + L)/2

def Monte_Carlo_1(runs, x, C):
    ## 1. Generate item sizes using Monte Carlo and the Box-Muller transform
    # Use initial x to determine the means, standard deviations, etc.
    idx = np.where(x==1)
    N = np.sum(x).astype(int)           # Number of items in knapsack
    
    # Parameters of the desired normal distribution
    mean = mu[idx]
    std = sigma[idx]
    unit_rev = unit_revenue[idx]

    
    # Generate 7 uniformly distributed random variables U
    U1 = np.array([uniform.rvs(size = runs) for i in range(N)])
    U2 = np.array([uniform.rvs(size = runs) for i in range(N)])
    
    
    # Box-Muller transform
    Z = np.sqrt(-2* np.log(U1)) * np.cos(2 * np.pi * U2)
    
    # Transform to desired normal distribution
    D = mean + Z.T * std

    
    ## 2. In cases where the simulation yields negative item sizes, these values can be adjusted to 0.
    D[np.where(D<0)] = 0 
    
    ## 3. Calculate the revenue, cost, and total profit.
    # Revenue per run
    revenue = np.matmul(D,unit_rev)
    
    # Salvage revenue per run
    remaining_capacity = C - np.sum(D, axis = 1)
    salvage_revenue = np.where( remaining_capacity > 0, s*remaining_capacity, 0)
    
    total_revenue = revenue + salvage_revenue
    
    # Underage costs
    cu = np.where( remaining_capacity < 0, p_prime*remaining_capacity, 0)
    
    # Capacity costs per run
    cc = C*c
    
    total_costs = cu + cc
    
    # Total profit per run
    total_profit = total_revenue - total_costs
    
    # (If needed) Averages
    avg_revenue = np.sum(total_revenue)/runs 
    avg_costs   = np.sum(total_costs)/runs
    avg_profit  = np.sum(total_profit)/runs
    
    # P_i = np.sum(D.T*(R-s), axis = 1) - (p-s)*(np.sum(D,axis = 0)-C)*(np.sum(D,axis = 0)>C) + s*C
    # P = np.sum(P_i)/runs
    
    return total_revenue, total_costs, total_profit


revenue_1, costs_1, profit_1 = Monte_Carlo_1(5120, initial_x, C)
np.sum(profit_1)/5120

#%% State the 95% confidence interval for the expected profit

def confidence_interval(profit):
    mean = np.mean(profit)
    std = np.std(profit)
    Z_alpha = 1.960
    return [mean - Z_alpha*std/np.sqrt(len(profit)), mean + Z_alpha*std/np.sqrt(len(profit)) ]

## Determining CI 
# Note that in order to half the confidence interval, one needs to use 4*n runs instead of n. 
revenue_1, costs_1, profit_1 = Monte_Carlo_1(5, initial_x, C) 
CI = confidence_interval(profit_1)
CI[1]-CI[0] # gap is about 30, so we need to half the interval at least 5 times to get a gap of 1. This means that we need 5120 runs. 

revenue_1, costs_1, profit_1 = Monte_Carlo_1(5120, initial_x, C) 
CI = confidence_interval(profit_1)
CI[1]-CI[0] # indeed the gap is smaller than 1. 


#%% Construct a histogram for the profit distribution based on your simulation results
plt.hist(profit_1, 50)
plt.show()

#%%
## Calculate the true objective function.
def loss_function_normal(C,mu,sigma):
    z = (C - mu)/np.sqrt(sigma)
    return np.sqrt(sigma)*(norm(0,1).pdf(z) - z + z*norm(0,1).cdf(z))
    
true_profit = np.matmul((unit_revenue[0:7]-s),mu[0:7])-p*loss_function_normal(C,np.sum(mu[0:7]),np.sum(sigma[0:7])) + s*C - c*C
true_profit

#%% Part 2.2
### greedy algorithm: if the cost is negative in 10 instances, add the next item.  
def Monte_Carlo_2_pebble(runs, p, s, C, revenue):
    L = 30 * 15
    U = 120 * 15
    x = [0 for _ in range(15)]
    item = np.argmax(revenue) # Select item with highest unit revenue
    x[item] = 1
    revenue[item] = -1 # Set revenue to -1, indicating that x[item] has been added to the knapsack. 
    go = 1
    prev_cost =100000000
    while go:
        ## 1. Generate item sizes
        # The first seven items are in the knapsack, so we only have to generate 7 item sizes
        D = np.array([norm.rvs(size = 10, loc = Data[0][i], scale = Data[1][i]) for i in range(15) if x[i] == 1])

        ## 2. In cases where the simulation yields negative item sizes, these values can be adjusted to 0.
        D[np.where(D<0)] = 0

        ## 3. Calculate the cost 
        total_size = np.sum(D, axis=0) # The item sizes of items together (total_size[i] is the total size of one run)
        remaining_capacity = C - total_size
        cost = np.where(remaining_capacity > 0, -s*remaining_capacity, p*remaining_capacity) # if the remaining capcity is positive, we use the salvage value otherwise the cost value p
        capacity_cost = C*c
        cost = np.array([cost[i] + capacity_cost for i in range(np.size(cost))])
        new_cost = max(cost)
        
        if new_cost > prev_cost or sum(x) == 15: #We are making costs or we added all items, so we stop
            go = 0
            if new_cost < prev_cost:
                x[item] = 0     #remove item from list, since previous solution was better. 

        if go:
            item = np.argmax(revenue) # Select next item with highest unit revenue
            x[item] = 1
            revenue[item] = -1 # Set revenue to -1, indicating that x[item] has been added to the knapsack. 
            
    print("The values of x:", x)
    
    D = np.array([norm.rvs(size = runs, loc = Data[0][i], scale = Data[1][i]) for i in range(15) if x[i] == 1])

    ## 2. In cases where the simulation yields negative item sizes, these values can be adjusted to 0.
    D[np.where(D<0)] = 0

    ## 3. Calculate the revenue, cost, and total profit.
    R = np.array(Data[2][0:15]) # unit revenue
    R = [R[i] for i in range(15) if x[i] == 1]
    revenue = np.matmul(D.T, R)  # total revenue (revenue[i] is the revenue of one run)
    total_size = np.sum(D, axis=0) # The item sizes of items together (total_size[i] is the total size of one run)
    remaining_capacity = C - total_size
    cost = np.where(remaining_capacity > 0, -s*remaining_capacity, p*remaining_capacity) # if the remaining capcity is positive, we use the salvage value otherwise the cost value p
    capacity_cost = C*c
    cost = [cost[i] + capacity_cost for i in range(np.size(cost))]
    profit = revenue - cost
    return revenue, cost, profit

rev, cost, profit = Monte_Carlo_2_pebble(10240, p_prime, s, C, unit_revenue)
CI = confidence_interval(profit)
CI[1]-CI[0] # indeed the gap is smaller than 1. 
np.mean(profit)
### plot histogram
plt.hist(profit, 50)
plt.show()

#%% Part 2.2 - Conduct a Monte Carlo simulation of the improved solution
### 1) Suggest an improved solutions
## Greedy algorithm based on the service level (z) (see Article)
# Initialization 
x = np.zeros(15)

r_i = (unit_revenue -s)*mu

# Finding the minimum service level
z_i = norm.ppf(1-r_i/(p*mu))
z = np.min(z_i) + 0.1

Phi_z = 1-norm.cdf(z)

# Measure pf attractiveness
rho = (r_i - p*mu*Phi_z)/sigma
sorted_indexes = np.argsort(rho)

# Logic regarding measure of attractiveness: if x_j = 1 and rho_k(z) > rho_j(z) then x_k = 1
# Similarly, if x_j = 0 and rho_k(x) < rho_j(z) then x_k = 0

## Add items according to attractiveness until previous capacity is reached
mu_new = np.array([mu[i] for i in sorted_indexes])
mu_cumsum = np.cumsum(mu_new)
in_set = np.where(mu_cumsum <= C, sorted_indexes, -1)
in_set_cor = np.delete(in_set, np.where(in_set == -1))

x[in_set_cor] = 1

## News vendor problem to solve for optimal capacity
r = np.average(unit_revenue*x)      # average revenue

cu = r - c + p_prime                # underage cost
co = c - s                          # overage cost

mu_D = np.sum(mu*x)
sigma_D = np.sum(sigma*x)

alpha = cu / (cu + co)
Z = norm.ppf(alpha)

C_star = mu_D + Z*np.sqrt(sigma_D)

# Resulting profit
revenue_I, costs_I, profit_I = Monte_Carlo_1(5120, x, C_star)
np.sum(profit_I)/5120

#%% 95% confidence interval
CI_I = confidence_interval(profit_I)

#%% Plot histogram improved solution
plt.hist(profit_I, 50)
plt.show()

#%% True profit improved solution
idx = np.where(x == 1)
true_profit_I = np.matmul((unit_revenue[idx]-s),mu[idx])-p*loss_function_normal(C,np.sum(mu[idx]),np.sum(sigma[idx])) + s*C - c*C
true_profit_I


#%%
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

#%%
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
    capacity_cost = C*c
    cost = [cost[i] + capacity_cost for i in range(len(cost))]
    profit = revenue - cost                                                           # NOTE: add cost for capacity
    return revenue, cost, profit
