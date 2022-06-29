import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.distances import directed_hausdorff, getHyperVolume, average_directed_haussdorf_distance, diameter
from utils.calc_pareto import get_pareto_undominated_by
from gpflow.utilities import print_summary

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination

import gpflow
import sobol_seq


class GaussianProcess(object):
    def __init__(self, d:int, lowerBound: float, upperBound: float, X = None, Y = None, noise_variance=0.01):
        self.d = d
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.X = X
        self.Y = Y
        self.noise_variance = noise_variance
        self.GPR : gpflow.models.GPR = None
        self.opt = gpflow.optimizers.Scipy()

    def print(self):
        print(self.__dict__)
        
    def addSample(self, x, y, save=False, filename=None):
        if self.X is None or self.Y is None:
            self.X = np.array([x])
            self.Y = np.array([y])
            return
        self.X = np.append(self.X, [x], axis=0)
        self.Y = np.append(self.Y, [y], axis=0)
        if save and filename is not None:
            self.writeSample(filename, x,y)

    def updateGP(self):
        self.GPR = gpflow.models.GPR(
                [self.X, self.Y],
                kernel = gpflow.kernels.SquaredExponential(), 
                #mean_function = gpflow.mean_functions.Constant(),
                noise_variance = self.noise_variance
            )

    def optimizeKernel(self):
        self.opt.minimize(
                self.GPR.training_loss, 
                variables = self.GPR.trainable_variables)
        
    ## EVALUATION OF RESULT
    def evaluateOptimum(self, optimum):
        """
        returns ->
            metrics :
                r_t : simple regret    : global optimum - actual optimum
                R_t : inference regret : global optimum - inferenced optimum
        """
        
        xx = sobol_seq.i4_sobol_generate(self.d,1_000)
        mean, _ = self.GPR.predict_y(xx)

        r_t = np.amin(self.Y)-optimum
        R_t = np.amin(mean)-optimum
        
        return {"r_t": r_t, "R_t": R_t}
        
    ## PLOT FUNCTIONS
    def plotSamples(self, n_samples=5):
        
        fig, axs = plt.subplots(nrows = 1, ncols=self.d)
        xx = np.linspace(self.lowerBound, self.upperBound, 100).reshape(100, 1)
        
        if self.d>1:
            for j in range(self.d):
                grid = np.zeros((100,self.d))
                grid[:,j]=xx[:,0]
                mean, var = self.GPR.predict_y(grid)
                samples = self.GPR.predict_f_samples(grid, n_samples)

                axs[j].plot(self.X[:,j], self.Y, 'kx', mew=2)
                axs[j].plot(grid[:,j], mean[:,0], 'C0', lw=2)
                axs[j].fill_between(grid[:,j],
                                mean[:,0] - 2*np.sqrt(var[:,0]),
                                mean[:,0] + 2*np.sqrt(var[:,0]),
                                color='C0', alpha=0.2)

                axs[j].plot(grid[:,j], samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
        else:
            
            mean, var = self.GPR.predict_y(xx)
            samples = self.GPR.predict_f_samples(xx, n_samples)
            axs.plot(self.X, self.Y, 'kx', mew=2)
            axs.plot(xx, mean[:,0], 'C0', lw=2)
            axs.fill_between(xx[:,0],
                            mean[:,0] - 2*np.sqrt(var[:,0]),
                            mean[:,0] + 2*np.sqrt(var[:,0]),
                            color='C0', alpha=0.2)

            axs.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
        plt.show()