"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
import tensorflow as tf

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import sobol_seq

from models.MESProblem import MESProblem
from models.GaussianProcess import GaussianProcess

mes_acq_hp = {
    'id' : 2,
    'name' : "MES",
    'help' : "This function develops mesmo procedure described by Belakaria(2020) with some upgrading changes.",
    'hps' : [
        {
            'name' : "N",
            'type': "%d",
            'help': "Number of points to generate samples",
            'default': 1_000
        },
        {
            'name' : "M",
            'type': "%d",
            'help': "Number of samples (optimums) to generate",
            'default': 50
        }
    ]
}

basic_mes_acq_hp = {
    'id' : 1,
    'name' : "basic MES",
    'help' : "This function develops mesmo procedure described by Belakaria(2020).",
    'hps' : [
        {
            'name' : 'N',
            'type': "%d",
            'help': "Number of points to generate samples",
            'default': 1_000
        }
    ]
}

def mes_acq(GP: GaussianProcess, N: int = 1_000, M: int = 50, showplots: bool = False):
    
    xx = sobol_seq.i4_sobol_generate(GP.d,N)
    samples = GP.multiGPR.predict_f_samples(xx,M)
    optimums = tf.math.reduce_min(samples, axis = 1)

    problem = MESProblem(GP, optimums)
    algorithm = NSGA2()
    res = minimize(problem, algorithm)

    if showplots:        
        x_tries, acqs = problem.curve() 
        GP.plotMES(res.X[0], res.F[0], x_tries, acqs)

    if (len(res.X.shape)>1):
        res_x = res.X[0]
    else:
        res_x = res.X

    return res_x, res.F[0]

def basic_mes_acq(GP: GaussianProcess, N: int = 1_000, M: int = 50,showplots: bool = False):
    xx = sobol_seq.i4_sobol_generate(GP.d,N)
    samples = GP.multiGPR.predict_y(xx)
    optimums = tf.math.reduce_min(samples, axis = 1)

    problem = MESProblem(GP, optimums)
    algorithm = NSGA2()
    res = minimize(problem, algorithm)

    if showplots:        
        x_tries, acqs = problem.curve() 
        GP.plotMES(res.X[0], res.F[0], x_tries, acqs)

    if (len(res.X.shape)>1):
        res_x = res.X[0]
    else:
        res_x = res.X

    return res_x, res.F[0]