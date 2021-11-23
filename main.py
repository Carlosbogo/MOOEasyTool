"""
@author: Miguel Taibo Martínez

Date: 16-Nov 2021
"""

import os
import numpy as np
import gpflow

from utils import blockPrint, enablePrint
from models.GaussianProcess import GaussianProcess
from acquisition_functions.MESMO import mesmo_acq
from arguments.arguments import MainArguments

### Definitions of outside parameters
from benchmark import f1,f2,f3

functions=[f1, f2]
constraints=[] 
def evaluation(x,d):
    y = [f(x,d) for f in functions]
    c = [f(x,d) for f in constraints]
    return np.array(y+c)

O = len(functions)
C = len(constraints)

### Definition of inside parameters
Aguments = MainArguments()
args = Aguments.parse()

if args.quiet:
    blockPrint()

outputFile = os.path.join(args.dir_path, args.output_file+'.csv')
if args.save:
    Aguments.writeArguments(outputFile)

d = args.d
    
seed = args.seed
np.random.seed(seed)

total_iter = args.total_iter
initial_iter = args.initial_iter

lowerBound = args.lower_bound
upperBound = args.upper_bound

### Kernerl configuration 
k = gpflow.kernels.SquaredExponential()
### GPs Initialization
GP = GaussianProcess(O, C, d, lowerBound, upperBound, k, noise_variance=2e-6)

if args.save:
    GP.writeGPHeader(outputFile)

# x_rand = np.zeros(d)
# y_rand = evaluation(x_rand,d)
# GP.addSample(x_rand,y_rand, args.save, outputFile)

### Initial samples, at least 1
for l in range(initial_iter):
    ## Get random evaluation point
    while True:
        x_rand = np.random.uniform(lowerBound, upperBound, d)
        if GP.X is None or not x_rand in GP.X:
            break
    ## EVALUATION OF THE OUTSIDE FUNCTION
    y_rand = evaluation(x_rand,d)
    GP.addSample(x_rand,y_rand, args.save, outputFile)

GP.updateGPR()
GP.optimizeKernel()
if args.showplots:
    GP.plotSamples()

for l in range(total_iter):
    
    ## Search of the best acquisition function
    x_best, acq_best = mesmo_acq(GP, showplots = args.showplots)

    ## EVALUATION OF THE OUTSIDE FUNCTION
    y_best = evaluation(x_best,d)
    
    GP.addSample(x_best,y_best, args.save, outputFile)      ## Add new sample to the model
    GP.updateGPR()                                          ## Update data on the GP regressor
    GP.optimizeKernel()                                     ## Optimize kernel hyperparameters


from models.GPProblem import GPProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.optimize import minimize

problem = GPProblem(GP)
algorithm = NSGA2()
termination = get_termination("n_gen", 40)
res = minimize(problem,
                algorithm,
                termination)
import matplotlib.pyplot as plt
plt.plot(res.X[:,0], res.X[:,1], 'o')
plt.show()
import pdb
pdb.set_trace()