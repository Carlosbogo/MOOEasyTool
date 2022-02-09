"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
import numpy as np

from pymoo.core.problem import Problem

class GPProblem(Problem):
    def __init__(self, GP):
        super().__init__(n_var=GP.d, n_obj=GP.O, n_constr=GP.C, xl=np.array(GP.lowerBounds), xu=np.array(GP.upperBounds))
        self.GPR = GP.GPR

    def _evaluate(self, x, out, *args, **kwargs):
        mean, _ = self.GPR.predict_y(np.array([[x]]))
        out["F"] = np.column_stack(mean[0])