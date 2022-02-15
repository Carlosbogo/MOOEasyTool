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

from MOObenchmark import MOOackley
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
evaluation = MOOackley

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

plt.plot(X, Z[:,0], 'b')
plt.plot(X, Z[:,1], 'k')
plt.plot(pareto_set, real_pareto[:,0], 'xr', markersize=5)
plt.plot(pareto_set, real_pareto[:,1], 'xr', markersize=5)
plt.show()

plt.plot(np.reshape(Z,(-1,2))[:,0], np.reshape(Z,(-1,2))[:,1], 'kx')
plt.plot(real_pareto[:,0], real_pareto[:,1], 'rx')
plt.show()



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

distancias = []
ds1 = []
ds2 = []

for l in range(total_iter):
    
    ## Search of the best acquisition function
    x_best, acq_best = mes_acq(GP, showplots = args.showplots)
    ## EVALUATION OF THE OUTSIDE FUNCTION
    y_best = evaluation(x_best)
    
    ## UPDATE
    GP.addSample(x_best,y_best, args.save, outputFile)      ## Add new sample to the model
    GP.updateGPR()                                          ## Update data on the GP regressor
    GP.optimizeKernel()                                     ## Optimize kernel hyperparameters

    pareto, distancia, d1, d2 = GP.evaluatePareto(real_pareto, showparetos = args.showparetos, saveparetos=args.saveparetos)
    distancias.append(distancia)
    ds1.append(d1)
    ds2.append(d2)
    

if args.save:
    idxs = [i for i,_ in enumerate(distancias)]

    plt.plot(idxs, distancias)
    plt.plot(idxs, ds1)
    plt.plot(idxs, ds2)
    plt.show()


    df = pd.DataFrame({'d': distancias, 'd1': ds1, 'd2':ds2})
    df.to_csv("./CSVs/"+args.savename+".csv")
