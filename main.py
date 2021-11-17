"""
@author: Miguel Taibo Martínez

Date: 16-Nov 2021
"""

from math import sqrt
import numpy as np
import sobol_seq
import gpflow
from gpflow.utilities import print_summary, set_trainable, to_default_float

from utils import blockPrint, enablePrint
from models.GaussianProcess import GaussianProcess
from acquisition_functions.MESMO import mesmo_acq
from arguments.arguments import MainArguments


### Definitions of outside parameters
def XSquared(x,d):
    res = 0
    for i in range(d):
        res+=x[i]*x[i]
    return res

def XRoot(x,d):
    res = 0
    for i in range(d):
        res+=sqrt(x[i])
    return res

def f1(x,d):
    res = 0
    for i in range(d):
        res+=(x[i]+0.5)*(x[i]+0.5)
        res-=7/12
    return res

def f2(x,d):
    res = 0
    for i in range(d):
        res+=(x[i]-0.5)*(x[i]-0.5)
        res-=7/12
    return res

functions=[f1,f2]
constraints=[] 
def evaluation(x,d):
    y = [f(x,d) for f in functions]
    c = [f(x,d) for f in constraints]
    return np.array(y+c)

O = len(functions)
C = len(constraints)


### Definition of inside parameters
args = MainArguments().parse()

if args.quiet:
    blockPrint()

d = args.d
    
seed = args.seed
np.random.seed(seed)

total_iter = args.total_iter
initial_iter = args.initial_iter

lowerBound = args.lower_bound
upperBound = args.upper_bound


grid = sobol_seq.i4_sobol_generate(d,1000,np.random.randint(0,1000))
bound_grid = np.vectorize(lambda x : x*(upperBound-lowerBound)+lowerBound)(grid)

### Kernerl configuration 
k = gpflow.kernels.SquaredExponential()
### GPs Initialization
GP = GaussianProcess(O, C, d, lowerBound, upperBound, k, noise_variance=2e-6)


### Initial samples, at least 1
for l in range(initial_iter):
    while True:
        index = np.random.choice(bound_grid.shape[0], 1)[0]  
        x_rand = bound_grid[index]
        if GP.X is None or not x_rand in GP.X:
            break
    y_rand = evaluation(x_rand,d)
    GP.addSample(x_rand,y_rand)

GP.updateGPR()
GP.optimizeKernel()
GP.plot()

for l in range(total_iter):
    
    ## GRID SEARCH OVER THE INPUT SPACE FOR THE OPTIMUM
    ## OF THE ACQUISITION FUNCTION
    x_tries = bound_grid
    acqs = mesmo_acq(x_tries, GP)

    sorted_index = np.argsort(acqs)

    x_best = x_tries[sorted_index[0]]

    for index in sorted_index:
        x_best = x_tries[index]
        if not x_best in GP.X:
            break

    GP.plotACQ(x_tries,acqs,x_best, acqs[sorted_index[0]])

    ## EVALUATION OF THE OUTSIDE FUNCTION
    y_best = evaluation(x_best,d)
    
    GP.addSample(x_best,y_best)     ## Add new sample to the model
    GP.updateGPR()                  ## Update data on the GP regressor
    GP.optimizeKernel()             ## Optimize kernel hyperparameters
    # plotGPR(GP)