"""
@author: Miguel Taibo Martínez

Date: 16-Nov 2021
"""

import os
import numpy as np
import gpflow
import matplotlib.pyplot as plt
import pandas as pd

from models.GaussianProcess import GaussianProcess
from acquisition_functions.MES import mes_acq
from acquisition_functions.PESMO import pesmo_acq
from acquisition_functions.MESMO import mesmo_acq
from arguments.arguments import MainArguments

from MOObenchmark import MOOackley, MOOquadratic
from utils.calc_pareto import get_pareto_undominated_by, getSetfromFront

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


### Definitions of outside parameters
def evaluation(x):
    return MOOquadratic(x, c1=-.5, c2=.5)

O = 2
C = 0

N = 10_001
X = np.linspace(args.lower_bound,args.upper_bound,N)
Z = np.zeros((N,2))

for i in range(N):
    Z[i]=evaluation(X[i])

real_pareto = get_pareto_undominated_by(np.reshape(Z,(-1,2)))
real_pareto = real_pareto[np.argsort(real_pareto[:,1])]
pareto_set = getSetfromFront(X,Z,real_pareto)

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
    y_rand = evaluation(x_rand)
    GP.addSample(x_rand,y_rand, args.save, outputFile)

GP.updateGPR()
GP.optimizeKernel()
if args.showplots:
    GP.plotSamples()

row = {
    'ns' : len(GP.X),
    'x'  : x_rand,
    'y'  : y_rand
}
metrics = GP.evaluatePareto(real_pareto, showparetos = False, saveparetos = True)
row.update(metrics)
df = pd.DataFrame({k: [v] for k, v in row.items()})

for l in range(total_iter):
    
    ## Search of the best acquisition function
    x_best, acq_best = mes_acq(GP, showplots = args.showplots)
    ## EVALUATION OF THE OUTSIDE FUNCTION
    y_best = evaluation(x_best)
    
    ## UPDATE
    GP.addSample(x_best,y_best, args.save, outputFile)      ## Add new sample to the model
    GP.updateGPR()                                          ## Update data on the GP regressor
    GP.optimizeKernel()                                     ## Optimize kernel hyperparameters

    ## Evaluate Pareto (distances and hypervolumes)
    row = {
        'ns' : len(GP.X),
        'x'  : x_best,
        'y'  : y_best
    }
    metrics = GP.evaluatePareto(real_pareto, showparetos = False, saveparetos = True)
    row.update(metrics)

    df = df.append(row, ignore_index = True)
    

if args.save:
    idxs = [i for i,_ in enumerate(distancias)]

    plt.plot(idxs, distancias)
    plt.plot(idxs, ds1)
    plt.plot(idxs, ds2)
    plt.show()

    df = pd.DataFrame({'d': distancias, 'd1': ds1, 'd2':ds2})
    df.to_csv("./CSVs/"+args.savename+".csv")
