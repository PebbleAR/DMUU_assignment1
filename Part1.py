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

def generate_mus(n):
    mus = []
    for _ in range(n):
        mus.append(generate_mu())
    return mus

mus = generate_mus(15)
mus

# Generating sigma
t_x = 2.0736
def Beta_pdf(x):
    60 * x**2 *(1-x)**3
    
def generate_sigma():
    while True:
        Y = uniform.rvs(size = 1)
        U = uniform.rvs(size = 1)
        if U <= Beta_pdf(Y)/t_x:
            X = Y
            break
    return X

sigma = []
for i in range(0,15):
    sigma.append(generate_sigma())

# Generating Unit revenue
def generate_uniform():
    U_1 = uniform.rvs(size = 1)
    U_2 = uniform.rvs(size = 1)
    return U_1, U_2

def generate_X(p1, p2):
    U = generate_uniform()
    if p1 < U[0][0]:
        X = -(1/2)*math.log(U[1][0])
    else:
        print(U[1][0])
        X = U[1][0]
    return X

n=15
p1 = 1/3
p2 = 2/3
R = []
for i in range(0,n):
    x = generate_X(p1, p2)
    if x < 5/2:
        r = 15+2*x
    else:
        r = 15+2*(5/2)
    R.append(r)
print(R)
    

#[150.15256811493003, 115.19677491162399, 111.67979915328472, 134.30643467186982, 126.56929525374949, 195.00577403381462, 172.5978737876039, 118.55164097302003, 136.05687557651135, 134.747749160613, 122.0422866146018, 162.73107320992452, 197.7589332244999, 169.49703379075925, 115.6649611292763]


# %%
