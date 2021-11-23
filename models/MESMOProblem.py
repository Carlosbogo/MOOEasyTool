"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
from math import sqrt
import numpy as np
from scipy.stats import norm

from pymoo.core.problem import Problem

from models.GaussianProcess import GaussianProcess

class MESMOProblem(Problem):
    def __init__(self, GP: GaussianProcess, maximums):
        super().__init__(n_var=GP.d, n_obj=GP.O, n_constr=GP.C, xl=np.array([GP.lowerBound]*GP.d), xu=np.array([GP.upperBound]*GP.d))
        self.GPR = GP.GPR
        self.maximums = maximums

    def _evaluate(self, X, out, *args, **kwargs):
        mean, var = self.GPR.predict_y(np.array([[X]]))

        varphi = np.divide(np.subtract(self.maximums,mean),np.sqrt(var))
        pdf, cdf = norm.pdf(varphi), np.maximum(norm.cdf(varphi),1e-30)
        acq = varphi*pdf / (2*cdf) - np.log(cdf)
        out["F"] = np.column_stack(np.sum(acq,axis=3)[0])