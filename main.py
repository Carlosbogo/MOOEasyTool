"""
@author: Miguel Taibo Martínez

Date: 16-Nov 2021
"""

import os
import numpy as np
import gpflow

from models.GaussianProcess import GaussianProcess
from acquisition_functions.MES import mes_acq
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

outputFile = os.path.join(args.dir_path, args.output_file+'.csv')
if args.save:
    Aguments.writeArguments(outputFile)

d = args.d
    
seed = args.seed
np.random.seed(seed)

total_iter = args.total_iter
initial_iter = args.initial_iter

lowerBounds = [args.lower_bound]*d
upperBounds = [args.upper_bound]*d

### Kernerl configuration 
k = gpflow.kernels.SquaredExponential()
### GPs Initialization
GP = GaussianProcess(O, C, d, lowerBounds, upperBounds, k, noise_variance=2e-6)

if args.save:
    GP.writeGPHeader(outputFile)

### Initial samples, at least 1
for l in range(initial_iter):
    ## Get random evaluation point
    while True:
        x_rand = np.random.uniform(lowerBounds[0], upperBounds[0], d)
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

