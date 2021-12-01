"""
@author: Miguel Taibo Martínez

Date: Dec 2021
"""
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import Hypervolume


def plotHV(res):
    approx_ideal = res.F.min(axis=0)
    approx_nadir = res.F.max(axis=0)

    n_evals = np.array([e.evaluator.n_eval for e in res.history])
    opt = np.array([e.opt[0].F for e in res.history])

    metric = Hypervolume(ref_point= res.history[0].opt[0].F,
                     norm_ref_point=False,
                     zero_to_one=True,
                     ideal=approx_ideal,
                     nadir=approx_nadir)

    hv = [metric.do(_F) for _F in opt]

    plt.figure(figsize=(7, 5))
    plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
    plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
    plt.title("Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.show()