"""
@author: Miguel Taibo Martínez

Date: Jan 2021
"""
import numpy as np
import tensorflow as tf
import sobol_seq
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem

from models.GaussianProcess import GaussianProcess
from utils.EPAlgorithm import EP

class EPProblem(ElementwiseProblem):
    def __init__(self, GP: GaussianProcess, Paretos):
        super().__init__(n_var=GP.d, n_obj=1, n_constr=0, xl=np.array(GP.lowerBounds), xu=np.array(GP.upperBounds))
        self.GPR = GP.GPR
        self.means, self.vars = [], []
        for pareto_set in Paretos:
            mean, var = self.GPR.predict_y(pareto_set)
            self.means.append(mean)
            self.vars.append(var)

    def _evaluate(self, x, out, *args, **kwargs):
        mean, var = self.GPR.predict_y(np.array([[x]]))

        pareto_var = tf.zeros(var[0][0].shape, dtype=tf.dtypes.float64)
        for means,vars in zip(self.means,self.vars):
            _, var_p = EP(mean[0][0], var[0][0], means, vars)
            pareto_var = pareto_var +var_p

        acquisition = -tf.math.reduce_prod(var-pareto_var/len(self.means))
        print(acquisition)
        out["F"] = np.column_stack([acquisition])


    def curve(self):
        grid = sobol_seq.i4_sobol_generate(self.n_var,100)
        bound_grid = np.vectorize(lambda x : x*(self.xu[0]-self.xl[0])+self.xl[0])(grid)
        mean, var = self.GPR.predict_y(bound_grid)

        pareto_vars = []
        for m_x, v_x in zip(mean, var):
            pareto_var = tf.zeros(v_x.shape, dtype=tf.dtypes.float64)
            for means,vars in zip(self.means,self.vars):
                _, var_p = EP(m_x, v_x, means, vars)
                pareto_var = pareto_var +var_p
            pareto_vars.append(pareto_var)

        acquisition = -tf.math.reduce_prod(var-np.array(pareto_vars)/len(self.means), axis=1)
        return bound_grid, acquisition
   