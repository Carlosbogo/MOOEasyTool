"""
@author: Miguel Taibo Martínez

Date: April 2022
"""
import string
import numpy as np
import tensorflow as tf

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination, get_sampling
from pymoo.optimize import minimize
from scipy.stats import norm
import sobol_seq

from models.UseMOProblem import UseMOProblem
from models.GaussianProcess import GaussianProcess

usemo_acq_hp = {
    'id' : 3,
    'name' : "UseMO",
    'help' : "This function develops UseMo procedure described by Syrine Belakaria(2020).",
    'hps' : [
        {
            'name' : 'function',
            'type': "%o",
            'help': "Function to use in the first stage",
            'options': ["ei", "pi", "ucb"] 
        },
        {
            'name' : 'N',
            'type': "%d",
            'help': "Number of points to generate samples",
            'default': 1_000
        }
    ]
}


def pi(mean, var, optimum, e = 0):
    z = (optimum+e-mean)/np.sqrt(var)
    return 1 - norm.cdf(z)

def ei(mean, var, optimum, e = 0):
    z = (optimum+e-mean)/np.sqrt(var)
    return (mean- optimum-e)*(1-norm.cdf(z))+np.sqrt(var)*norm.pdf(z)

def ucb(mean, var, optimum, beta = 0.1, e=0):
    return mean + beta*var

def mes(mean, var, optimums, e = 0):
    acq = 0
    for optimum in optimums:
        z = (optimum+e-mean)/np.sqrt(var)
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)
        acq += -np.log(cdf)+z*pdf/2/cdf
    return acq/len(optimums)

def codeToFunction(code):
    if code=="pi":
        return pi
    elif code=="ei":
        return ei
    elif code=="ucb":
        return ucb
    elif code=="mes":
        return mes
    return None

def usemo_acq(GP: GaussianProcess, function : string, M: int = 5, N: int = 1_000, showplots: bool = False):

    xx = sobol_seq.i4_sobol_generate(GP.d,N)
    if function=="mes":
        samples = GP.multiGPR.predict_f_samples(xx,M)
        optimums = tf.math.reduce_min(samples, axis = 1)
    else:
        mean, _ = GP.multiGPR.predict_y(xx)
        optimums = tf.math.reduce_min(mean, axis = 0)

    problem = UseMOProblem(GP, codeToFunction(function), optimums)
    algorithm = NSGA2()
    res = minimize(problem,
                   algorithm,
                   save_history=True,
                   verbose=False)

    if showplots:
        x_tries, acqs = problem.curve() 
        GP.plotMES(res.X, res.F[0], x_tries, acqs)
        # for pareto in Paretos:
        #     GP.plotADF(res.X, pareto)

    _, var = GP.multiGPR.predict_y(res.X)

    return res.X[tf.argmax(tf.reduce_prod(var, axis=1))], res.F[tf.argmax(tf.reduce_prod(var, axis=1))]