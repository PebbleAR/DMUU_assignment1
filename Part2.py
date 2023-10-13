from Part1 import generate_data
import numpy as np
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt

np.random.seed(1)

# Generate data
Data = generate_data(15)
mu = np.array(Data[0])
sigma = np.array(Data[1])
unit_revenue = np.array(Data[2])

n       = 15          # Number of items
p_prime = 25          # Unit overflow cost
s       = 4           # Salvage value
c       = 15          # Unit capacity cost
p       = p_prime - s

initial_x = np.append(np.ones(7),np.zeros(8))

N = np.sum(initial_x).astype(int)

## Table 2, page 476 on Capacity level
L = 30 * n
U = 120 * n
C = (U + L)/2

# Monte Carlo Simulation of the initial solution
def Monte_Carlo_1(runs, x, C, D0 = None):
    ## Generate item sizes using Monte Carlo and the Box-Muller transform
    # Use initial x to determine the means, standard deviations, etc.
    idx = np.where(x==1)
    # N = np.sum(x).astype(int)         # Number of items in knapsack
    N = len(x)

    # Parameters of the desired normal distribution
    # mean = mu[idx]
    # std = sigma[idx]
    # unit_rev = unit_revenue[idx]

    mean = mu
    std = sigma
    unit_rev = unit_revenue
    
    D = D0

    # Generate D if it is not yet generated
    if D is None:
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
    revenue = np.matmul(D, unit_rev * x)
    
    # Salvage revenue per run
    remaining_capacity = C - np.matmul(D, x)
    salvage_revenue = np.where(remaining_capacity > 0, s*remaining_capacity, 0)   
    
    total_revenue = revenue + salvage_revenue
    
    # Underage costs
    cu = np.where(remaining_capacity < 0, p_prime*remaining_capacity, 0)
    
    # Capacity costs per run
    cc = C*c
    total_costs = cu + cc
    
    # Total profit per run
    total_profit = total_revenue - total_costs
    
    return total_revenue, total_costs, total_profit, D

# Monte Carlo Simulation of the improved solution
## Greedy algorithm based on the service level (z) (see Article)
def improved_solution(data, D):
    mu = np.array(data[0])
    sigma = np.array(data[1])
    unit_revenue = np.array(data[2])

    x = np.zeros(15)

    r_i = (unit_revenue-s)*mu

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
    revenue_I, costs_I, profit_I, D = Monte_Carlo_1(3225, x, C_star, D)

    return revenue_I, costs_I, profit_I, C_star, x 


# The number of runs is decided such that the gap of the confidence interval is smaller than 50
# This is the case when the number of runs is greater than 3225
revenue_1, costs_1, profit_1, D = Monte_Carlo_1(3225, initial_x, C)
revenue_I, costs_I, profit_I, C_star, x_I = improved_solution(Data, D)

# 95% confidence interval calculated
def confidence_interval(profit):
    mean = np.mean(profit)
    std = np.std(profit)
    Z_alpha = 1.960
    return [mean - Z_alpha*std/np.sqrt(len(profit)), mean + Z_alpha*std/np.sqrt(len(profit)) ]

CI = confidence_interval(profit_1)
CI_I = confidence_interval(profit_I)
print(f"The confidence interval of the initial solution with 3225 runs is: {CI}")
print(f"The confidence interval of the improved solution with 3225 runs is: {CI_I}")

# Histogram of the profit distributions
# initial solution
plt.hist(profit_1, 50)
plt.title("Histogram of Monte Carlo simulation of Initial solution")
plt.xlabel("Profit")
plt.show()

# improved solution
plt.hist(profit_I, 50)
plt.title("Histogram of Monte Carlo simulation of Improved solution")
plt.xlabel("Profit")
plt.show()

# True objective function
# standard normal loss function
def loss_function_normal(C,mu,sigma):
    z = (C - mu)/np.sqrt(sigma)
    return np.sqrt(sigma)*(norm(0,1).pdf(z) - z + z*norm(0,1).cdf(z))

# True profit initial solution    
true_profit = np.matmul((unit_revenue[0:7]-s),mu[0:7])-p_prime*loss_function_normal(C,np.sum(mu[0:7]),np.sum(sigma[0:7])) + s*C - c*C
print(f"The true objective of the initial simulation: {true_profit}")

# True profit improved solution
idx = np.where(x_I == 1)
true_profit_I = np.matmul((unit_revenue[idx]-s),mu[idx])-p_prime*loss_function_normal(C_star,np.sum(mu[idx]),np.sum(sigma[idx])) + s*C_star - c*C_star
print(f"The true objective of the improved simulation: {true_profit_I}")

# Confidence interval for the difference of the mean profits
def CI_diff_mean_profits(runs, data):
    diff_profits = []
    for _ in range(runs):
        sol_initial = Monte_Carlo_1(3225, initial_x, C)
        mean_prof_1 = np.mean(sol_initial[2])

        sol_improved = improved_solution(data, sol_initial[3])
        mean_prof_I = np.mean(sol_improved[2])

        diff_profits.append(abs(mean_prof_I - mean_prof_1))

    return confidence_interval(diff_profits)

CI_diff_mean_prof = CI_diff_mean_profits(10, Data)
print(f"The confidence interval of the difference of the mean profits with 10 runs is: {CI_diff_mean_prof}")

# Variance reduction

print(C_star)