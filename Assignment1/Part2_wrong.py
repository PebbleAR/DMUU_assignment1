from Part1 import generate_data
import numpy as np
from scipy.stats import norm

data = generate_data(15) #data[0]: mu, data[1]: sigma, data[2]: unit revenue
n = 15
x = np.ones(8)
y = np.zeros(8)
x = np.append(x,y)
L = 30 * n      # table 2 article
U = 120 * n     # table 2 article
C = U           # read page 464 for argumentation of chosen values for the constants
x[0] = (L+U)/(2*(U-L))
p = 25
s = 4
c = 15

print(x) #x_0 is additional item
m = 5120
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

profit = np.array(profit)
np.mean(profit)