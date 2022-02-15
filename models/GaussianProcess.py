"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.distances import directed_hausdorff

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination

import gpflow
import sobol_seq

from utils.ADFAlgorithm import ADF
from utils.EPAlgorithm import EP
from models.GPProblem import GPProblem

class GaussianProcess(object):
    def __init__(self, O:int, C:int, d:int, lowerBounds: float, upperBounds: float, kernel, X = None, Y = None, noise_variance=0.01):
        self.O = O
        self.C = C
        self.d = d
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.kernel = kernel
        self.X = X
        self.Y = Y
        self.noise_variance = noise_variance
        self.opt = gpflow.optimizers.Scipy()
        self.GPR : gpflow.models.GPR = None

    def addSample(self, x, y, save=False, filename=None):
        if self.X is None or self.Y is None:
            self.X = np.array([x])
            self.Y = np.array([y])
            return
        self.X = np.append(self.X, [x], axis=0)
        self.Y = np.append(self.Y, [y], axis=0)
        if save and filename is not None:
            self.writeSample(filename, x,y)

    def updateGPR(self):
        self.GPR = gpflow.models.GPR(
            [self.X, self.Y],
            kernel = self.kernel, 
            noise_variance = self.noise_variance)

    def optimizeKernel(self):
        self.opt.minimize(
            self.GPR.training_loss, 
            variables=self.GPR.trainable_variables)

    ## Saving into file methods
    def writeGPHeader(self,filename):
        vars  = ['x'+str(i) for i in range(self.d)]
        vars += ['y'+str(i) for i in range(self.O)]
        vars += ['c'+str(i) for i in range(self.C)]
        header = ''
        for v in vars:
            header += (v + ",") 
            
        file = open(filename,"a+")
        file.write(header[0:-1]+'\n')
        file.close()

    def writeSample(self, filename, x, y):
        vars  = [str(e) for e in x]
        vars += [str(e) for e in y]
        row   = ''
        for v in vars:
            row += (v+',')

        file = open(filename,"a+")
        file.write(row[0:-1]+'\n')
        file.close()

    ## Visualization methods
    def plot(self):

        fig, axs = plt.subplots(nrows = self.O, ncols=self.d)
        xx = np.linspace(self.lowerBounds[0], self.upperBounds[0], 100).reshape(100, 1)

        if self.d >1:
            for j in range(self.d):
                grid = np.zeros((100,self.d))
                grid[:,j]=xx[:,0]
                mean, var = self.GPR.predict_y(grid)

                for i in range(self.O):
                    axs[i, j].plot(self.X[:,j], self.Y[:,i], 'kx', mew=2)
                    axs[i, j].plot(grid[:,j], mean[:,i], 'C0', lw=2)
                    axs[i, j].fill_between(grid[:,j],
                                    mean[:,i] - 2*np.sqrt(var[:,i]),
                                    mean[:,i] + 2*np.sqrt(var[:,i]),
                                    color='C0', alpha=0.2)
        else:
            mean, var = self.GPR.predict_y(xx)
            for i in range(self.O):
                axs[i].plot(self.X, self.Y[:,i], 'kx', mew=2)
                axs[i].plot(xx[:,0], mean[:,i], 'C0', lw=2)
                axs[i].fill_between(xx[:,0],
                                mean[:,i] - 2*np.sqrt(var[:,i]),
                                mean[:,i] + 2*np.sqrt(var[:,i]),
                                color='C0', alpha=0.2)
        plt.show()

    def plotSamples(self, n_samples=5):
        fig, axs = plt.subplots(nrows = self.O, ncols=self.d)
        xx = np.linspace(self.lowerBounds[0], self.upperBounds[0], 100).reshape(100, 1)

        if self.d >1:
            for j in range(self.d):
                grid = np.zeros((100,self.d))
                grid[:,j]=xx[:,0]
                mean, var = self.GPR.predict_y(grid)
                samples = self.GPR.predict_f_samples(grid, n_samples)

                for i in range(self.O):
                    axs[i, j].plot(self.X[:,j], self.Y[:,i], 'kx', mew=2)
                    axs[i, j].plot(grid[:,j], mean[:,i], 'C0', lw=2)
                    axs[i, j].fill_between(grid[:,j],
                                    mean[:,i] - 2*np.sqrt(var[:,i]),
                                    mean[:,i] + 2*np.sqrt(var[:,i]),
                                    color='C0', alpha=0.2)

                    axs[i, j].plot(grid[:,j], samples[:, :, i].numpy().T, "C0", linewidth=0.5)


        else:
            mean, var = self.GPR.predict_y(xx)
            samples = self.GPR.predict_f_samples(xx, n_samples)
            
            for i in range(self.O):
                axs[i].plot(self.X, self.Y[:,i], 'kx', mew=2)
                axs[i].plot(xx[:,0], mean[:,i], 'C0', lw=2)
                axs[i].fill_between(xx[:,0],
                                mean[:,i] - 2*np.sqrt(var[:,i]),
                                mean[:,i] + 2*np.sqrt(var[:,i]),
                                color='C0', alpha=0.2)
        
                axs[i].plot(xx[:,0], samples[:, :, i].numpy().T, "C0", linewidth=0.5)
        plt.show()

    def plotMES(self, x_best, acq, x_tries, acqs):
        fig, axs = plt.subplots(nrows = self.O+1, ncols=self.d)
        xx = np.linspace(self.lowerBounds[0], self.upperBounds[0], 100).reshape(100, 1)

        if self.d >1:
            for j in range(self.d):
                grid = np.zeros((100,self.d))
                grid[:,j]=xx[:,0]
                mean, var = self.GPR.predict_y(grid)

                for i in range(self.O):
                    axs[i, j].plot(self.X[:,j], self.Y[:,i], 'kx', mew=2)
                    axs[i, j].plot(grid[:,j], mean[:,i], 'C0', lw=2)
                    axs[i, j].fill_between(grid[:,j],
                                    mean[:,i] - 2*np.sqrt(var[:,i]),
                                    mean[:,i] + 2*np.sqrt(var[:,i]),
                                    color='C0', alpha=0.2)
                        
                    if i==0:
                        axs[i, j].xaxis.set_label_position('top')
                        axs[i, j].set_xlabel("x"+str(j))
                    if j==0:
                        axs[i, j].set_ylabel("y"+str(i))
                    # axs[i, j].axvline(x=x_best[j], color='r')
                axs[self.O, j].plot(x_tries[:,j],acqs,'o', markersize=1)
                axs[self.O, j].plot([x_best[j]], [acq],'or', markersize=4)
                axs[self.O, j].set_ylim(acq-0.2, acq+2.2)

        else:
            mean, var = self.GPR.predict_y(xx)
            for i in range(self.O):
                axs[i].plot(self.X, self.Y[:,i], 'kx', mew=2)
                axs[i].plot(xx[:,0], mean[:,i], 'C0', lw=2)
                axs[i].fill_between(xx[:,0],
                                mean[:,i] - 2*np.sqrt(var[:,i]),
                                mean[:,i] + 2*np.sqrt(var[:,i]),
                                color='C0', alpha=0.2)
            
                if i==0:
                    axs[i].xaxis.set_label_position('top')
                    axs[i].set_xlabel("x0")
                axs[i].set_ylabel("y"+str(i))
            
            axs[self.O].plot(x_tries[:,0],acqs,'o', markersize=1)
            axs[self.O].plot([x_best[0]], [acq],'or', markersize=4)
            axs[self.O].set_ylim(acq-0.2, acq+2.2)
                    
        plt.show()    

    def plotADF(self, x_best, pareto):
        mean, var = self.GPR.predict_y(np.array([x_best]))

        mean_p, var_p = ADF(mean, var, pareto)
        fig, ax = plt.subplots()

        ax.plot(pareto[:,0], pareto[:,1], 'o', markersize=1)
        rect = patches.Rectangle((mean[0][0]-np.sqrt(var[0][0]), mean[0][1]-np.sqrt(var[0][1])), np.sqrt(var[0][0]), np.sqrt(var[0][0]), linewidth=2, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        rect = patches.Rectangle((mean_p[0][0]-np.sqrt(var_p[0][0]), mean_p[0][1]-np.sqrt(var_p[0][1])), np.sqrt(var_p[0][0]), np.sqrt(var_p[0][0]), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        plt.show()
    
    def plotEP(self, x_best, pareto):
        mean, var = self.GPR.predict_y(np.array([x_best]))
        means, vars = self.GPR.predict_y(np.array(pareto))
        mean_p, var_p = EP(mean, var, means, vars)

        fig, axs = plt.subplots(2)

        axs[0].plot(pareto[:,0], pareto[:,1], 'o', markersize=1)
        axs[0].plot(x_best[0], x_best[1], 'ro', markersize=1)

        axs[1].plot(means[:,0],means[:,1],'o',markersize=1)
        rect = patches.Rectangle((mean[0][0]-np.sqrt(var[0][0]), mean[0][1]-np.sqrt(var[0][1])), np.sqrt(var[0][0]), np.sqrt(var[0][0]), linewidth=2, edgecolor='k', facecolor='none')
        axs[1].add_patch(rect)

        rect = patches.Rectangle((mean_p[0][0]-np.sqrt(var_p[0][0]), mean_p[0][1]-np.sqrt(var_p[0][1])), np.sqrt(var_p[0][0]), np.sqrt(var_p[0][0]), linewidth=1, edgecolor='r', facecolor='none')
        axs[1].add_patch(rect)
        axs[1].set_xlim(-2,2)
        axs[1].set_ylim(-2,2)
        plt.show()

        import pdb
        pdb.set_trace()

    def plotObjectives(self, x_best, acq, x_tries):
        fig, axs = plt.subplots(nrows = self.O+1, ncols=self.d)
        xx = sobol_seq.i4_sobol_generate(self.d,1_000)

        mean, var = self.GPR.predict_y(xx)
        axs.plot(mean[:,0], mean[:,1], 'C0', lw=2)
        axs.fill_between(mean[:,0],
                mean[:,0] - 2*np.sqrt(var[:,0]),
                mean[:,0] + 2*np.sqrt(var[:,0]),
                color='C0', alpha=0.2)

        y_best, _ = self.GPR.predict_y(np.array([x_best]))
        axs.plot(y_best[0][0], y_best[0][1])

        plt.show()

    def plotACQS(self, x_tries, acqs,x_best, acq):
        fig, axs = plt.subplots(nrows = self.O+1, ncols=self.d)
        xx = np.linspace(self.lowerBounds[0], self.upperBounds[0], 100).reshape(100, 1)

        if self.d >1:
            for j in range(self.d):
                grid = np.zeros((100,self.d))
                grid[:,j]=xx[:,0]
                mean, var = self.GPR.predict_y(grid)

                for i in range(self.O):
                    axs[i, j].plot(self.X[:,j], self.Y[:,i], 'kx', mew=2)
                    axs[i, j].plot(grid[:,j], mean[:,i], 'C0', lw=2)
                    axs[i, j].fill_between(grid[:,j],
                                    mean[:,i] - 2*np.sqrt(var[:,i]),
                                    mean[:,i] + 2*np.sqrt(var[:,i]),
                                    color='C0', alpha=0.2)

                axs[self.O, j].plot(x_tries[:,j],acqs,'o', markersize=1)
                axs[self.O, j].plot([x_best[j]], [acq],'or', markersize=4)
                axs[self.O, j].set_ylim(acq-0.2, acq+2.2)

        else:
            mean, var = self.GPR.predict_y(xx)
            for i in range(self.O):
                axs[i].plot(self.X, self.Y[:,i], 'kx', mew=2)
                axs[i].plot(xx[:,0], mean[:,i], 'C0', lw=2)
                axs[i].fill_between(xx[:,0],
                                mean[:,i] - 2*np.sqrt(var[:,i]),
                                mean[:,i] + 2*np.sqrt(var[:,i]),
                                color='C0', alpha=0.2)
        
            axs[self.O].plot(x_tries[:,0],acqs,'o', markersize=1)
            axs[self.O].plot([x_best[0]], [acq],'or', markersize=4)
            axs[self.O].set_ylim(acq-0.2, acq+2.2)
            
        plt.show()

    def evaluatePareto(self, reference = None, showparetos: bool = False, saveparetos: bool = False):

        problem = GPProblem(self)
        res = minimize(problem,
                NSGA2(),
                get_termination("n_gen", 40),
                save_history=True,
                verbose=False)

        (distancia1,i1,j1) = directed_hausdorff(res.F, reference)
        # print("distancia1", distancia1)
        (distancia2,i2,j2) = directed_hausdorff(reference, res.F)
        # print("distancia2", distancia2)
          
        s1 = np.array([res.F[i1], reference[j1]])
        s2 = np.array([reference[i2], res.F[j2]])
        plt.plot(s1[:,0], s1[:,1], 'b', label="d1  "+ str(distancia1))
        plt.plot(s2[:,0], s2[:,1], 'b', label="d2  " + str(distancia2))
        
        F = res.F[np.argsort(res.F[:,1])]
        plt.plot(F[:,0],F[:,1], 'k', label='Pareto front')
        plt.plot(reference[:,0],reference[:,1], 'r', label='Pareto real')
        plt.legend()

        if saveparetos:  
            plt.savefig("ImagesExp/"+str(len(self.X))+'.png')
        if showparetos:
            plt.show()
        plt.clf()

        return res.F, distancia1 if distancia1>distancia2 else distancia2, distancia1, distancia2
