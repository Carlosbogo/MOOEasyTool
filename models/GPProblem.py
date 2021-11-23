"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
import numpy as np

from pymoo.core.problem import Problem

from models.GaussianProcess import GaussianProcess

class GPProblem(Problem):
    def __init__(self, GP: GaussianProcess):
        super().__init__(n_var=GP.d, n_obj=GP.O, n_constr=GP.C, xl=np.array([GP.lowerBound]*GP.d), xu=np.array([GP.upperBound]*GP.d))
        self.GPR = GP.GPR

    def _evaluate(self, X, out, *args, **kwargs):
        mean, _ = self.GPR.predict_y(np.array([[X]]))
        out["F"] = np.column_stack(mean[0][:,:,0:self.n_obj])
        out["G"] = np.column_stack(mean[0][:,:,self.n_obj: self.n_obj+self.n_constr])