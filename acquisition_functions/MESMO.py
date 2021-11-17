"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
from math import sqrt
from scipy.stats import norm
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
import numpy as np
from models.GaussianProcess import GaussianProcess
from pymoo.factory import get_termination
from pymoo.optimize import minimize

class GPProblem(Problem):
    def __init__(self, GP: GaussianProcess):
        super().__init__(n_var=GP.d, n_obj=GP.O, n_constr=GP.C, xl=np.array([GP.lowerBound]), xu=np.array([GP.upperBound]))
        self.GPR = GP.GPR

    def _evaluate(self, x, out, *args, **kwargs):
        mean, _ = self.GPR.predict_y(np.array([[x]]))
        out["F"] = np.column_stack(mean.numpy().tolist()[0])

def mesmo_acq(x_tries,GP: GaussianProcess):

    ## Compute pareto front
    problem = GPProblem(GP)
    algorithm = NSGA2()
    termination = get_termination("n_gen", 40)
    res = minimize( problem,
                    algorithm,
                    termination)
    maximums = [min(res.F[:,i]) for i in range(GP.O)]

    ## Apply mesmoc acquisition function
    mean, var = GP.GPR.predict_y(x_tries)
    acqs = []
    for m,v in zip(mean.numpy(),var.numpy()):

        acq = 0
        for i in range(GP.O):
            varphi = (maximums[i]-m[i]) / sqrt(v[i])
            pdf, cdf = norm.pdf(varphi), max(norm.cdf(varphi),1e-30)
            acq+= varphi*pdf / (2*cdf) - np.log(cdf)
        
        acqs.append(acq)

    return acqs

