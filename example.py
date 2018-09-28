import numpy as np
from steadystate import getSteadyStateDist

# a row-stochastic three-state Markov transition probability matrix
P = np.array([[0.9,0.07,0.03],[0.1,0.8,0.1],[0.15,0.25,0.6]])


q = getSteadyStateDist(P)

# we can show that this is the correct steady state distribution
# by showing that qP = q, i.e., q is the eigenvector with eigenvalue 1

print('The steady state distribution is')
print(q)
print('Verifying by calcluating qP, the result is')
print(np.matmul(q,P))
