"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
import numpy as np

from pymoo.core.problem import Problem

from models.GaussianProcess import GaussianProcess

class GPProblem(Problem):
    def __init__(self, GP: GaussianProcess):
        super().__init__(n_var=GP.d, n_obj=GP.O, n_constr=GP.C, xl=np.array([GP.lowerBound]), xu=np.array([GP.upperBound]))
        self.GPR = GP.GPR

    def _evaluate(self, x, out, *args, **kwargs):
        mean, _ = self.GPR.predict_y(np.array([[x]]))
        out["F"] = np.column_stack(mean.numpy().tolist()[0])