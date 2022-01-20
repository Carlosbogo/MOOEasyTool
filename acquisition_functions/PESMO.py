"""
@author: Miguel Taibo Martínez

Date: Juan 2021
"""
import pdb
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination, get_sampling
from pymoo.optimize import minimize

from models.EPProblem import EPProblem
from models.GaussianProcess import GaussianProcess
from utils.ParetoSample import getParetoFrontSamples, getParetoSetSamples, getParetoSamples

pesmo_acq_hp = {
    'id' : 4,
    'name' : "PESMO",
    'help' : "This function develops pesmo procedure described by Fernandez-Sanchez(2016).",
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

def pesmo_acq(GP: GaussianProcess, N: int = 1_000, M: int = 5, showplots: bool = False):

    Pareto_sets, Pareto_fronts = getParetoSamples(GP, N, M)

    problem = EPProblem(GP, np.array(Pareto_sets))
    algorithm = NSGA2()
    termination = get_termination("n_eval", 10)
    res = minimize(problem,
                   algorithm,
                   termination,
                   save_history=True,
                   verbose=False)

    if showplots or True:
        # x_tries, acqs = problem.curve() 
        # GP.plotMES(res.X, res.F[0], x_tries, acqs)
        for pareto in Pareto_sets:
            GP.plotEP(res.X, pareto)

    return res.X, res.F[0]