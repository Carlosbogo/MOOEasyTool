"""
@author: Miguel Taibo Martínez

Date: 16-Nov 2021
"""

import os
import numpy as np
import gpflow

from models.GaussianProcess import GaussianProcess
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

d = 1
    
seed = 10
np.random.seed(seed)

initial_iter = 2

lowerBound = -1
upperBound = 1

### Kernerl configuration 
k = gpflow.kernels.SquaredExponential()
### GPs Initialization
GP = GaussianProcess(O, C, d, lowerBound, upperBound, k, noise_variance=2e-6)

### Initial samples, at least 1
for l in range(initial_iter):
    ## Get random evaluation point
    while True:
        x_rand = np.random.uniform(lowerBound, upperBound, d)
        if GP.X is None or not x_rand in GP.X:
            break
    ## EVALUATION OF THE OUTSIDE FUNCTION
    y_rand = evaluation(x_rand,d)
    GP.addSample(x_rand,y_rand, False)

GP.updateGPR()
GP.optimizeKernel()

xx = np.linspace(lowerBound, upperBound, 1001).reshape(1001, 1)
samples = GP.GPR.predict_f_samples(xx, 10000)

mins = np.amin(samples,axis=1)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows =1, ncols=2)

axs[0].hist(mins[:,0], 100, density=True)
axs[1].hist(mins[:,1], 100, density=True)
plt.title("MES: y* distribution")
plt.show()

import pdb

def get_pareto(pts):
    pts = pts[np.argsort(pts[:,0])]
    res = [pts[0]]
    current_min = pts[0][1]
    for p in pts:
        if p[1]<current_min:
            current_min = p[1]
            res.append(p)
    return np.array(res)

pareto = np.array([])
first = True
for sample in samples:
    if first:
        pareto = get_pareto(sample.numpy())
        first = False
        continue
    pareto = np.append(pareto, get_pareto(sample.numpy()),axis=0)

plt.hist2d(pareto[:,0], pareto[:,1], bins=50,)
plt.xlabel("y0")
plt.ylabel("y1")
plt.colorbar()
plt.show()


pdb.set_trace()


from scipy.stats import norm
import numpy as np
R = 2
x = np.linspace(0,2,21)
y = np.sqrt(R**2-x**2)

[norm.pdf(x[i])*norm.pdf(y[i]) for i in range(len(x))]

