"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""

import numpy as np
import matplotlib.pyplot as plt

from models.GaussianProcess import GaussianProcess

def plotGPR(gp: GaussianProcess):
    xx = np.linspace(gp.lowerBound, gp.upperBound, 100).reshape(100, 1)
    mean, var = gp.GPR.predict_y(xx)
    fig, axs = plt.subplots(gp.O)
    for i in range(gp.O):
        axs[i].plot(gp.X, gp.Y[:,i], 'kx', mew=2)
        axs[i].plot(xx[:,0], mean[:,i], 'C0', lw=2)
        axs[i].fill_between(xx[:,0],
                        mean[:,i] - 2*np.sqrt(var[:,i]),
                        mean[:,i] + 2*np.sqrt(var[:,i]),
                        color='C0', alpha=0.2)
    plt.show()


def plotACQ(gp: GaussianProcess, x_tries, acqs,x_best, acq):
    xx = np.linspace(gp.lowerBound, gp.upperBound, 100).reshape(100, 1)
    mean, var = gp.GPR.predict_y(xx)
    fig, axs = plt.subplots(gp.O+1)
    for i in range(gp.O):
        axs[i].plot(gp.X, gp.Y[:,i], 'kx', mew=2)
        axs[i].plot(xx[:,0], mean[:,i], 'C0', lw=2)
        axs[i].fill_between(xx[:,0],
                        mean[:,i] - 2*np.sqrt(var[:,i]),
                        mean[:,i] + 2*np.sqrt(var[:,i]),
                        color='C0', alpha=0.2)
    
    axs[gp.O].plot(x_tries,acqs,'o', markersize=1)
    axs[gp.O].plot([x_best], [acq],'or', markersize=4)
    axs[gp.O].set_ylim(acq-0.2, acq+2.2)
    plt.show()
