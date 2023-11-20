import numpy as np
from scipy.stats import expon, uniform
import matplotlib.pyplot as plt
np.random.seed(1)

M = 1000000 # Number of samples for the waiting time
mu = 2
lamb = 1
rho = lamb / mu
###############################################################
### Exponential service times and uniform interarrival times ##
###############################################################

# Sample exponential service times
B = expon.rvs(loc=0, scale=1/mu, size=M)

# Sample uniform interarrival times
a, b = 0, 2/lamb  # choices of a and b, s.t. E[A] = 1/5 and thus rho = 5/6. 
A = uniform.rvs(loc = a, scale = b - a, size = M ) # Sample from Uniform(a,b)

# Lindley's recursion for waiting times
W = [0]
for i in range(0,M-1):
    W.append(max(W[i] + B[i] - A[i], 0))

# Numeric approximation
numeric = np.mean(W)

# Determine Kingman's approximation 
# coefficient of variation of uniform distribution
c_a = 2/np.sqrt(12)
# coefficient of variation of exponential distribution
c_s = 1

kingman = rho/(1-rho) * (c_a**2 + c_s**2)/2 * 1/mu
print("Uniform interarrivals, exponential service times")
print(f"numeric approximation: {numeric}, Kingman's approximation: {kingman}")

###############################################################
### Uniform service times and Exponential interarrival times ##
###############################################################

# Sample exponential interarrival times
A = expon.rvs(loc=0, scale=1/lamb, size=M)

# Sample uniform service times
a, b = 0, 2/mu  # choices of a and b, s.t. E[B] = 1/6 and thus rho = 5/6. 
B = uniform.rvs(loc = a, scale = b - a, size = M ) # Sample from Uniform(a,b)

# Lindley's recursion for waiting times
W = [0] # first entry is 0 as we start from an empty system
for i in range(0,M-1):
    W.append(max(W[i] + B[i] - A[i], 0))

# Numeric approximation
numeric = np.mean(W)

# Determine Kingman's approximation 
# coefficient of variation of uniform distribution
c_s = 2/np.sqrt(12)
# coefficient of variation of exponential distribution
c_a = 1

kingman = rho/(1-rho) * (c_a**2 + c_s**2)/2 * 1/mu
print("Exponential interarrivals, uniform service times")
print(f"numeric approximation: {numeric}, Kingman's approximation: {kingman}")

###############################################################
### Uniform service times and Uniform interarrival times ##
###############################################################

# Sample uniform interarrival times
c, d = 0, 2/lamb  # choices of a and b, s.t. E[A] = 1/5. 
A = uniform.rvs(loc = c, scale = d - c, size = M ) # Sample from Uniform(a,b)

# Sample uniform service times
a, b = 0, 2/mu  # choices of a and b, s.t. E[B] = 1/6 and thus rho = 5/6. 
B = uniform.rvs(loc = a, scale = b - a, size = M ) # Sample from Uniform(a,b)

# Lindley's recursion for waiting times
W = [0] # first entry is 0 as we start from an empty system
for i in range(0,M-1):
    W.append(max(W[i] + B[i] - A[i], 0))

# Numeric approximation
numeric = np.mean(W)

# Determine Kingman's approximation 
# coefficient of variation of uniform distribution
c_s = 2/np.sqrt(12)
# coefficient of variation of exponential distribution
c_a = 2/np.sqrt(12) 

kingman = rho/(1-rho) * (c_a**2 + c_s**2)/2 * 1/mu
print("Uniform interarrivals, uniform service times")
print(f"numeric approximation: {numeric}, Kingman's approximation: {kingman}")

###############################################################
### Deterministic service times and uniform interarrival times ##
###############################################################

# Sample deterministic service times
B = np.ones(M)*(1/mu)

# Sample uniform interarrival times
a, b = 0, 2/lamb # choices of a and b, s.t. E[A] = 1/5 and thus rho = 5/6. 
A = uniform.rvs(loc = a, scale = b - a, size = M ) # Sample from Uniform(a,b)

# Lindley's recursion for waiting times
W = [0]
for i in range(0,M-1):
    W.append(max(W[i] + B[i] - A[i], 0))

# Numeric approximation
numeric = np.mean(W)

# Determine Kingman's approximation 
# coefficient of variation of uniform distribution
c_a = 2/np.sqrt(12)
# coefficient of variation of discrete distribution
c_s = 0

kingman = rho/(1-rho) * (c_a**2 + c_s**2)/2 * 1/mu
print("Uniform interarrivals, deterministic service times")
print(f"numeric approximation: {numeric}, Kingman's approximation: {kingman}")

###############################################################
### Deterministic service times and Exponential interarrival times ##
###############################################################

# Sample exponential interarrival times
A = expon.rvs(loc=0, scale=1/lamb, size=M)

# Sample deterministic service times
B = np.ones(M)*(1/mu)

# Lindley's recursion for waiting times
W = [0] # first entry is 0 as we start from an empty system
for i in range(0,M-1):
    W.append(max(W[i] + B[i] - A[i], 0))

# Numeric approximation
numeric = np.mean(W)

# Determine Kingman's approximation 
# coefficient of variation of deterministic distribution
c_s = 0
# coefficient of variation of exponential distribution
c_a = 1

kingman = rho/(1-rho) * (c_a**2 + c_s**2)/2 * 1/mu
print("exponential interarrivals, deterministic service times")
print(f"numeric approximation: {numeric}, Kingman's approximation: {kingman}")

###############################################################
### Deterministic service times and deterministic interarrival times ##
###############################################################

# Sample deterministic interarrival times
A = np.ones(M)*(1/lamb)

# Sample deterministic service times
B = np.ones(M)*(1/mu)

# Lindley's recursion for waiting times
W = [0] # first entry is 0 as we start from an empty system
for i in range(0,M-1):
    W.append(max(W[i] + B[i] - A[i], 0))

# Numeric approximation
numeric = np.mean(W)

# Determine Kingman's approximation 
# coefficient of variation of deterministic distribution
c_s = 0
# coefficient of variation of exponential distribution
c_a = 0

kingman = rho/(1-rho) * (c_a**2 + c_s**2)/2 * 1/mu
print("Deterministic interarrivals, deterministic service times")
print(f"numeric approximation: {numeric}, Kingman's approximation: {kingman}") # These values could also have been computed exact

###############################################################
### Uniform service times and Deterministic interarrival times ##
###############################################################

# Sample deterministic interarrival times
A = np.ones(M)*(1/lamb)

# Sample uniform service times
a, b = 0, 2/mu  # choices of a and b, s.t. E[B] = 1/6 and thus rho = 5/6. 
B = uniform.rvs(loc = a, scale = b - a, size = M ) # Sample from Uniform(a,b)

# Lindley's recursion for waiting times
W = [0] # first entry is 0 as we start from an empty system
for i in range(0,M-1):
    W.append(max(W[i] + B[i] - A[i], 0))

# Numeric approximation
numeric = np.mean(W)

# Determine Kingman's approximation 
# coefficient of variation of uniform distribution
c_s = 2/np.sqrt(12)
# coefficient of variation of exponential distribution
c_a = 0

kingman = rho/(1-rho) * (c_a**2 + c_s**2)/2 * 1/mu
print("Deterministic interarrivals, uniform service times")
print(f"numeric approximation: {numeric}, Kingman's approximation: {kingman}")

###############################################################
### Exponential service times and deterministic interarrival times ##
###############################################################

# Sample exponential service times
B = expon.rvs(loc=0, scale=1/mu, size=M)

# Sample deterministic interarrival times
A = np.ones(M)*(1/lamb)

# Lindley's recursion for waiting times
W = [0]
for i in range(0,M-1):
    W.append(max(W[i] + B[i] - A[i], 0))

# Numeric approximation
numeric = np.mean(W)

# Determine Kingman's approximation 
# coefficient of variation of deterministic distribution
c_a = 0
# coefficient of variation of exponential distribution
c_s = 1

kingman = rho/(1-rho) * (c_a**2 + c_s**2)/2 * 1/mu
print("Deterministic interarrivals, exponential service times")
print(f"numeric approximation: {numeric}, Kingman's approximation: {kingman}")