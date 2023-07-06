"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
import numpy as np
from numpy.random.mtrand import normal
from scipy.stats import norm
import sobol_seq
import tensorflow as tf

from pymoo.core.problem import Problem

from models.GaussianProcess import GaussianProcess

class MESProblem(Problem):
    def __init__(self, GP: GaussianProcess, optimums):
        super().__init__(n_var=GP.d, n_obj=GP.O, n_constr=GP.C, xl=np.array(GP.lowerBounds), xu=np.array(GP.upperBounds))
        self.multiGPR = GP.multiGPR
        self.optimums = optimums

    def _evaluate(self, X, out, *args, **kwargs):
        mean, var = self.multiGPR.predict_y(np.array([[X]]))
        acq = tf.zeros_like(mean)
        for optimum in self.optimums:
            varphi = (optimum-mean)/tf.math.sqrt(var)
            pdf, cdf = norm.pdf(varphi), tf.math.maximum(norm.cdf(varphi),1e-30)
            acq += varphi*pdf / (2*cdf) - tf.math.log(cdf)
        out["F"] = np.column_stack(tf.math.reduce_sum(acq,axis=3)[0])


    def curve(self):

        grid = sobol_seq.i4_sobol_generate(self.n_var,1000)
        bound_grid = np.vectorize(lambda x : x*(self.xu[0]-self.xl[0])+self.xl[0])(grid)
        
        mean, var = self.multiGPR.predict_y(bound_grid)

        acq = tf.zeros_like(mean)
        for optimum in self.optimums:
            varphi = (optimum-mean)/tf.math.sqrt(var)
            pdf, cdf = norm.pdf(varphi), tf.math.maximum(norm.cdf(varphi),1e-30)
            acq += varphi*pdf / (2*cdf) - tf.math.log(cdf)
        return bound_grid, tf.math.reduce_sum(acq,axis=1)