import numpy as np
from scipy.stats import binom, uniform, poisson
import matplotlib.pyplot as plt
import math

np.random.seed(1)

# Generating mu
def generate_mu():
    Y = 0
    while not 1 <= Y <= 10:
        U = uniform.rvs(size = 1)
        Y = -10 * np.log(U)
    mu = 100 + 10 * Y
    return mu[0]

# Generating sigma
t_x = 2.0736

def Beta_pdf(x):
    return 60 * x**2 *(1-x)**3
    
def generate_sigma():
    while True:
        Y = uniform.rvs(size = 1)
        U = uniform.rvs(size = 1)
        if U <= Beta_pdf(Y)/t_x:
            X = Y
            break
    return X[0]

# Generating Unit revenue
p1 = 1/3
p2 = 1 - p1

def generate_uniform():
    U_1 = uniform.rvs(size = 1)
    U_2 = uniform.rvs(size = 1)
    return U_1, U_2

def generate_X(p1):
    U = generate_uniform()
    if p1 < U[0][0]:
        X = -(1/2)*math.log(U[1][0])
    else:
        X = U[1][0]
    return X

def generate_R():
    x = generate_X(p1)
    if x < 5/2:
        r = 15+2*x
    else:
        r = 15+2*(5/2)
    return r

# Generate data
n = 15
def generate_data(n):
    mus, sigmas, R = [], [], []
    for _ in range(n):
        mus.append(generate_mu())
        sigmas.append(generate_sigma())
        R.append(generate_R())
    return mus, sigmas, R

data = generate_data(n)

print(data[0])
print()
print(data[1])
print()
print(data[2])