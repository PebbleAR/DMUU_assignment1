import numpy as np
from scipy.stats import binom, uniform, poisson
import matplotlib.pyplot as plt
import math

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

#[150.15256811493003, 115.19677491162399, 111.67979915328472, 134.30643467186982, 126.56929525374949, 195.00577403381462, 172.5978737876039, 118.55164097302003, 136.05687557651135, 134.747749160613, 122.0422866146018, 162.73107320992452, 197.7589332244999, 169.49703379075925, 115.6649611292763]

