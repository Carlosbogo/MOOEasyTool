"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination, get_sampling
from pymoo.optimize import minimize

from models.ADFProblem import ADFProblem
from models.GaussianProcess import GaussianProcess
from utils.ParetoSample import getParetoFrontSamples

mesmo_acq_hp = {
    'id' : 3,
    'name' : "MESMO",
    'help' : "This function develops mesmo procedure described by Fernandez-Sanchez(2020).",
    'hps' : [
        {
            'name' : 'N',
            'type': "%d",
            'help': "Number of points to generate samples",
            'default': 1_000
        },
        {
            'name' : 'M',
            'type': "%d",
            'help': "Number of samples (Pareto fronts) to generate",
            'default': 50
        }
    ]
}

def mesmo_acq(GP: GaussianProcess, N: int = 1_000, M: int = 5, showplots: bool = False):

    Paretos = getParetoFrontSamples(GP, N, M)

    problem = ADFProblem(GP, np.array(Paretos))
    algorithm = NSGA2()
    termination = get_termination("n_gen", 10)
    res = minimize(problem,
                   algorithm,
                   termination,
                   save_history=True,
                   verbose=False)

    if showplots:
        x_tries, acqs = problem.curve() 
        GP.plotMES(res.X, res.F[0], x_tries, acqs)
        # for pareto in Paretos:
        #     GP.plotADF(res.X, pareto)

    return res.X, res.F[0]