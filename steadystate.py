import numpy as np

# If P is a square row-stochastic matrix, returns the 
# normalized (sum 1) eigenvector associated with the 
# unit eigenvalue.
#
# For Markov chains with state transition probability 
# matrix P, this vector gives the steady state 
# distribution over the states.
#
# If P does not have a unit eigenvalue, returns None.
# However, sensible results are not guaranteed for
# arbitrary matrices.

def getSteadyStateDist(P):
    
    tolerance = 1e-10
    
    # need the left eigenvectors
    [u,v] = np.linalg.eig(np.transpose(P))
    v = np.transpose(v)
    
    index = 0
    for i in u:
        if np.abs(i - 1.) < tolerance:
            return np.real(v[index,:] / np.sum(v[index,:]))
        index += 1
        
    return None 


