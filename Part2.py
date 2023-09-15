from Part1 import generate_data
import numpy as np

data = generate_data(15) #data[0]: mu, data[1]: sigma, data[2]: unit revenue
n = 15
x = np.ones(8)
y = np.zeros(8)
x = np.append(x,y)
L = 30 * n      # table 2 article
U = 120 * n     # table 2 article
C = U           # read page 464 for argumentation of chosen values for the constants
x[0] = L+U/(2*(U-L))

print(x) #x_0 is additional item)