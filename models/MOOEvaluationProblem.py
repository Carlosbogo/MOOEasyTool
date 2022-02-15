"""
@author: Miguel Taibo Martínez

Date: Febrero 2022
"""
import numpy as np
from numpy.random.mtrand import normal
from scipy.stats import norm
import sobol_seq
import tensorflow as tf

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import Problem

from models.GaussianProcess import GaussianProcess

class MOOEvaluationProblem(ElementwiseProblem):
    """
        Encuentra el pareto front de un problema MOO definido a partir de una funcion
        con 1D de entrada y 2 objetivos
    """
    def __init__(self, evaluation, lowerBound, upperBound):
        super().__init__(n_var=1, n_obj=2, n_constr=0, xl=np.array([lowerBound]), xu=np.array([upperBound]))
        self.evaluation = evaluation

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.column_stack(self.evaluation(X))
