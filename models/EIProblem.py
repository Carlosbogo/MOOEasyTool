"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
from math import sqrt
import numpy as np
from scipy.stats import norm
import sobol_seq

from pymoo.core.problem import Problem

from models.GaussianProcess import GaussianProcess

class EIProblem(Problem):
    def __init__(self, GP: GaussianProcess, maximums):
        super().__init__(n_var=GP.d, n_obj=GP.O, n_constr=GP.C, xl=np.array([GP.lowerBound]*GP.d), xu=np.array([GP.upperBound]*GP.d))
        self.GPR = GP.GPR
        self.maximums = maximums

    def _evaluate(self, X, out, *args, **kwargs):
        mean, var = self.GPR.predict_y(np.array([[X]]))
        
        varphi = np.divide(np.subtract(self.maximums,mean[:,:,:,0:self.n_obj]),np.sqrt(var[:,:,:,0:self.n_obj]))
        pdf, cdf = norm.pdf(varphi), np.maximum(norm.cdf(varphi),1e-30)
        acq = (mean[:,:,:,0:self.n_obj] - self.maximums)*(1-cdf)+np.sqrt(var)*pdf
        out["F"] = np.column_stack(-acq[0])
        out["G"] = np.column_stack(mean[0][:,:,self.n_obj: self.n_obj+self.n_constr])

    def curve(self):
        grid = sobol_seq.i4_sobol_generate(self.n_var,1000,np.random.randint(0,1000))
        bound_grid = np.vectorize(lambda x : x*(self.xu[0]-self.xl[0])+self.xl[0])(grid)
        mean, var = self.GPR.predict_y(bound_grid)

        varphi = np.divide(np.subtract(self.maximums,mean[:,0:self.n_obj]),np.sqrt(var[:,0:self.n_obj]))
        pdf, cdf = norm.pdf(varphi), np.maximum(norm.cdf(varphi),1e-30)
        acq = (mean[:,0:self.n_obj] - self.maximums)*(1-cdf)+np.sqrt(var)*pdf
        return bound_grid, np.sum(acq,axis=1)