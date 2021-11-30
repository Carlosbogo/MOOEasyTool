"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
import numpy as np
import tensorflow as tf
import sobol_seq
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem

from models.GaussianProcess import GaussianProcess
from utils.ADFAlgorithm import ADF

class ADFProblem(Problem):
    def __init__(self, GP: GaussianProcess, Paretos):
        super().__init__(n_var=GP.d, n_obj=1, n_constr=0, xl=np.array([GP.lowerBound]*GP.d), xu=np.array([GP.upperBound]*GP.d))
        self.GPR = GP.GPR
        self.Paretos = Paretos

    def _evaluate(self, x, out, *args, **kwargs):
        mean, var = self.GPR.predict_y(np.array([[x]]))

        pareto_var = tf.zeros(var[0][0].shape, dtype=tf.dtypes.float64)
        for pareto in self.Paretos:
            _, var_p = ADF(mean[0][0], var[0][0], pareto)
            pareto_var = pareto_var +var_p
            
        acquisition = -tf.math.reduce_prod(var-pareto_var/len(self.Paretos), axis=3)[0]
        out["F"] = np.column_stack(acquisition)


    def curve(self):
        grid = sobol_seq.i4_sobol_generate(self.n_var,100)
        bound_grid = np.vectorize(lambda x : x*(self.xu[0]-self.xl[0])+self.xl[0])(grid)
        mean, var = self.GPR.predict_y(bound_grid)

        pareto_var = tf.zeros(var[0][0].shape, dtype=tf.dtypes.float64)
        for pareto in self.Paretos:
            _, var_p = ADF(mean, var, pareto)
            var_p = tf.where(tf.math.is_nan(var_p), var, var_p)
            pareto_var = pareto_var +var_p

        acquisition = -tf.math.reduce_prod(var-pareto_var/len(self.Paretos), axis=1)
        return bound_grid, acquisition
   