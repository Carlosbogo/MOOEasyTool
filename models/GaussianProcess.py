"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""

import numpy as np
import gpflow

class GaussianProcess(object):
    def __init__(self, O:int, C:int, d:int, lowerBound: float, upperBound: float, kernel, X = None, Y = None, noise_variance=0.01):
        self.O = O
        self.C = C
        self.d = d
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.kernel = kernel
        self.X = X
        self.Y = Y
        self.noise_variance = noise_variance
        self.opt = gpflow.optimizers.Scipy()
        self.GPR = None

    def addSample(self, x, y):
        if self.X is None or self.Y is None:
            self.X = np.array([x])
            self.Y = np.array([y])
            return
        self.X = np.append(self.X, [x], axis=0)
        self.Y = np.append(self.Y, [y], axis=0)

    def updateGPR(self):
        self.GPR = gpflow.models.GPR(
            [self.X, self.Y],
            kernel= self.kernel, 
            noise_variance=self.noise_variance)

    def optimizeKernel(self):
        self.opt.minimize(
            self.GPR.training_loss, 
            variables=self.GPR.trainable_variables)
