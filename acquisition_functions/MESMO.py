"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
import numpy as np
import tensorflow as tf
import sobol_seq

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination, get_sampling
from pymoo.optimize import minimize

from models.ADFProblem import ADFProblem
from models.GaussianProcess import GaussianProcess
from utils.calc_pareto import get_pareto_frontier, get_pareto_undominated_by

def mesmo_acq(GP: GaussianProcess, N: int = 1_000, M: int = 5, showplots: bool = False):

    Paretos = []
    xx = sobol_seq.i4_sobol_generate(GP.d,1_000)
    samples = GP.GPR.predict_f_samples(xx,M)
    maxs = np.amax(np.amax(samples, axis=0),axis=0)
    for sample in samples:
        for o, gold_medal in enumerate(sample[np.argmin(sample, axis=0)]):
            new_sample = sobol_seq.i4_sobol_generate(GP.O-1,100)*(np.append(maxs[:o], maxs[o+1:])-np.append(gold_medal[:o], gold_medal[o+1:]))+np.append(gold_medal[:o], gold_medal[o+1:])
            new_sample = np.insert(new_sample, o, gold_medal[o]*np.ones(new_sample.shape[0]), axis=1)
            sample = np.append(sample, new_sample, axis=0)


        pareto = get_pareto_undominated_by(sample)
        Paretos.append(pareto)

    problem = ADFProblem(GP, np.array(Paretos))
    algorithm = NSGA2(pop_size=50)
    termination = get_termination("n_gen", 4)
    res = minimize(problem,
            algorithm,
            termination,
            save_history=True,
            verbose=False)

    if showplots:
        x_tries, acqs = problem.curve() 
        GP.plotMES(res.X, res.F[0], x_tries, acqs)
        for pareto in Paretos:
            GP.plotADF(res.X, pareto)

    return res.X, res.F[0]