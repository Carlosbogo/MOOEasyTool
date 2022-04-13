"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
import numpy as np
from numpy.random.mtrand import normal
from scipy.stats import norm
import sobol_seq
import tensorflow as tf

from pymoo.core.problem import Problem

from models.GaussianProcess import GaussianProcess

class UseMOProblem(Problem):
    def __init__(self, GP: GaussianProcess, function, optimums):
        super().__init__(n_var=GP.d, n_obj=GP.O, n_constr=GP.C, xl=np.array(GP.lowerBounds), xu=np.array(GP.upperBounds))
        self.function = function
        self.optimums = optimums
        self.multiGPR = GP.multiGPR

    def _evaluate(self, X, out, *args, **kwargs):
        mean, var = self.multiGPR.predict_y(np.array([[X]]))

        # import pdb
        # pdb.set_trace()
        out["F"] = np.column_stack(self.function(mean, var, self.optimums)[0])

    def curve(self):

        grid = sobol_seq.i4_sobol_generate(self.n_var,1000)
        bound_grid = np.vectorize(lambda x : x*(self.xu[0]-self.xl[0])+self.xl[0])(grid)
        
        mean, var = self.multiGPR.predict_y(bound_grid)

        return bound_grid, self.function(mean, var, self.optimums)[0]

import matplotlib.pyplot as plt

def plotMV(xx, mean, var):
    fig, axs = plt.subplots(mean.shape[-1])
    for i in range(mean.shape[-1]):
        axs[i].plot(xx[:,0], mean[:,i], 'xb', lw=2)
        axs[i].plot(xx[:,0], var[:,i], 'xr', lw=2)

    plt.show()