import numpy as np
from scipy.stats import uniform
import math

np.random.seed(1)

# Generate from uniform distribution
def generate_uniform(size):
    return uniform.rvs(size = size)

# Generating mu
def generate_mu():
    Y = 0
    while not 1 <= Y <= 10:
        U = generate_uniform(1)
        Y = -10 * np.log(U)
    mu = 100 + 10 * Y
    return mu[0]

# Generating sigma
t_x = 2.0736

def Beta_pdf(x):
    return 60 * x**2 *(1-x)**3

def generate_sigma():
    while True:
        Y, U = generate_uniform(2)
        if U < Beta_pdf(Y)/t_x:
            X = Y
            sigma = 10 + 20 * X
            break
    return sigma

# Generating Unit revenue
p1 = 1/3
p2 = 1 - p1

def generate_X(p1):
    U1, U2 = generate_uniform(2)
    if U1 < p1:
        X = -(1/2)*math.log(U2)
    else:
        X = U2
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
    return [mus, sigmas, R]

data = generate_data(n)