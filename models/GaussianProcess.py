"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.distances import directed_hausdorff, getHyperVolume, average_directed_haussdorf_distance, diameter
from utils.calc_pareto import get_pareto_undominated_by
from gpflow.utilities import print_summary

from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination

import gpflow
import sobol_seq

from utils.ADFAlgorithm import ADF
from utils.EPAlgorithm import EP
from models.GPProblem import GPProblem

class GaussianProcess(object):
    def __init__(self, O:int, C:int, d:int, lowerBounds: float, upperBounds: float, X = None, Y = None, noise_variance=0.01):
        self.O = O
        self.C = C
        self.d = d
        self.lowerBounds = lowerBounds
        self.upperBounds = upperBounds
        self.X = np.array(X, dtype=float)
        self.Y = np.array(Y, dtype=float)
        self.noise_variance = noise_variance
        self.multiGPR : MultiGPR = None

    def addSample(self, x, y, save=False, filename=None):
        if self.X is None or self.Y is None:
            self.X = np.array([x])
            self.Y = np.array([y])
            return
        print("Current x = ", self.X)
        print("New x = ", x)
        self.X = np.append(self.X, [x], axis=0)
        self.Y = np.append(self.Y, [y], axis=0)
        if save and filename is not None:
            self.writeSample(filename, x,y)

    def updateGP(self):
        self.multiGPR = MultiGPR(X = np.array(self.X, dtype=float), Y = self.Y, noise_variance = self.noise_variance)

    def optimizeKernel(self):
        self.multiGPR.optimizeKernel()

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
                mean, var = self.multiGPR.predict_y(grid)

                for i in range(self.O):
                    axs[i, j].plot(self.X[:,j], self.Y[:,i], 'kx', mew=2)
                    axs[i, j].plot(grid[:,j], mean[:,i], 'C0', lw=2)
                    axs[i, j].fill_between(grid[:,j],
                                    mean[:,i] - 2*np.sqrt(var[:,i]),
                                    mean[:,i] + 2*np.sqrt(var[:,i]),
                                    color='C0', alpha=0.2)
        else:
            mean, var = self.multiGPR.predict_y(xx)
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
                mean, var = self.multiGPR.predict_y(grid)
                samples = self.multiGPR.predict_f_samples(grid, n_samples)

                for i in range(self.O):
                    axs[i, j].plot(self.X[:,j], self.Y[:,i], 'kx', mew=2)
                    axs[i, j].plot(grid[:,j], mean[:,i], 'C0', lw=2)
                    axs[i, j].fill_between(grid[:,j],
                                    mean[:,i] - 2*np.sqrt(var[:,i]),
                                    mean[:,i] + 2*np.sqrt(var[:,i]),
                                    color='C0', alpha=0.2)

                    axs[i, j].plot(grid[:,j], samples[:, :, i].numpy().T, "C0", linewidth=0.5)


        else:
            mean, var = self.multiGPR.predict_y(xx)
            samples = self.multiGPR.predict_f_samples(xx, n_samples)
            
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
                mean, var = self.multiGPR.predict_y(grid)

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
            mean, var = self.multiGPR.predict_y(xx)
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
        mean, var = self.multiGPR.predict_y(np.array([x_best]))

        mean_p, var_p = ADF(mean, var, pareto)
        fig, ax = plt.subplots()

        ax.plot(pareto[:,0], pareto[:,1], 'o', markersize=1)
        rect = patches.Rectangle((mean[0][0]-np.sqrt(var[0][0]), mean[0][1]-np.sqrt(var[0][1])), np.sqrt(var[0][0]), np.sqrt(var[0][0]), linewidth=2, edgecolor='k', facecolor='none')
        ax.add_patch(rect)

        rect = patches.Rectangle((mean_p[0][0]-np.sqrt(var_p[0][0]), mean_p[0][1]-np.sqrt(var_p[0][1])), np.sqrt(var_p[0][0]), np.sqrt(var_p[0][0]), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # ax.set_xlim(-2,2)
        # ax.set_ylim(-2,2)
        plt.show()
    
    def plotEP(self, x_best, pareto):
        mean, var = self.multiGPR.predict_y(np.array([x_best]))
        means, vars = self.multiGPR.predict_y(np.array(pareto))
        mean_p, var_p = EP(mean[0], var[0], means, vars)

        fig, axs = plt.subplots(2)
    
        if self.d>1:
            axs[0].plot(pareto[:,0], pareto[:,1], 'o', markersize=1)
            axs[0].plot(x_best[0], x_best[1], 'ro', markersize=1)
        else:
            axs[0].plot(pareto, [1 for _ in pareto], 'o', markersize=1)
            axs[0].plot(x_best, [1], 'o', markersize=1)
        axs[1].plot(means[:,0],means[:,1],'o',markersize=1)
        rect = patches.Rectangle((mean[0][0]-np.sqrt(var[0][0]), mean[0][1]-np.sqrt(var[0][1])), np.sqrt(var[0][0]), np.sqrt(var[0][0]), linewidth=2, edgecolor='k', facecolor='none')
        axs[1].add_patch(rect)

        rect = patches.Rectangle((mean_p[0][0]-np.sqrt(var_p[0][0]), mean_p[0][1]-np.sqrt(var_p[0][1])), np.sqrt(var_p[0][0]), np.sqrt(var_p[0][0]), linewidth=1, edgecolor='r', facecolor='none')
        axs[1].add_patch(rect)
        plt.show()

    def plotObjectives(self, x_best, acq, x_tries):
        fig, axs = plt.subplots(nrows = self.O+1, ncols=self.d)
        xx = sobol_seq.i4_sobol_generate(self.d,1_000)

        mean, var = self.multiGPR.predict_y(xx)
        axs.plot(mean[:,0], mean[:,1], 'C0', lw=2)
        axs.fill_between(mean[:,0],
                mean[:,0] - 2*np.sqrt(var[:,0]),
                mean[:,0] + 2*np.sqrt(var[:,0]),
                color='C0', alpha=0.2)

        y_best, _ = self.multiGPR.predict_y(np.array([x_best]))
        axs.plot(y_best[0][0], y_best[0][1])

        plt.show()

    def plotACQS(self, x_tries, acqs,x_best, acq):
        fig, axs = plt.subplots(nrows = self.O+1, ncols=self.d)
        xx = np.linspace(self.lowerBounds[0], self.upperBounds[0], 100).reshape(100, 1)

        if self.d >1:
            for j in range(self.d):
                grid = np.zeros((100,self.d))
                grid[:,j]=xx[:,0]
                mean, var = self.multiGPR.predict_y(grid)

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
            mean, var = self.multiGPR.predict_y(xx)
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

    def evaluatePareto(self, pareto_real = None, showparetos: bool = False, saveparetos: bool = False):

        ## Computation of Pareto Estimated
        problem = GPProblem(self)
        res = minimize(problem,
                NSGA2(),
                get_termination("n_gen", 40),
                save_history=True,
                verbose=False)
        pareto_estimated = res.F

        ## Computation of best known pareto
        best_known_pareto = get_pareto_undominated_by(self.Y)

        ## Computation of input space distances
        (d_current_previous, _, _)  = directed_hausdorff(self.Y[-1], self.Y[:-1])
        (xd_current_previous, _, _) = directed_hausdorff(self.X[-1], self.X[:-1])

        ### Computation of all 6 distances
        (d_e_r, i_e_r, j_e_r) = directed_hausdorff(pareto_estimated, pareto_real)
        (d_r_e, i_r_e, j_r_e) = directed_hausdorff(pareto_real, pareto_estimated)
          
        (d_e_k, i_e_k, j_e_k) = directed_hausdorff(pareto_estimated, best_known_pareto)
        (d_k_e, i_k_e, j_k_e) = directed_hausdorff(best_known_pareto, pareto_estimated)

        (d_k_r, i_k_r, j_k_r) = directed_hausdorff(best_known_pareto, pareto_real)
        (d_r_k, i_r_k, j_r_k) = directed_hausdorff(pareto_real, best_known_pareto)

        ### Plot distances
        def plotDistance(pareto1, pareto2, i, j, name, value):
            plt.arrow(  pareto1[i,0], pareto1[i,1], 
                        pareto2[j,0]-pareto1[i,0], pareto2[j,1]-pareto1[i,1],
                        head_width  = (abs(pareto2[j,0]-pareto1[i,0]) + abs(pareto2[j,1]-pareto1[i,1]))/30,
                        head_length = (abs(pareto2[j,0]-pareto1[i,0]) + abs(pareto2[j,1]-pareto1[i,1]))/20, 
                        length_includes_head = True,
                        label=name+"  "+ str(value))

        plotDistance(pareto_estimated, pareto_real, i_e_r, j_e_r, "d_e_r", d_e_r)
        plotDistance(pareto_real, pareto_estimated, i_r_e, j_r_e, "d_r_e", d_r_e)
        plotDistance(pareto_estimated, best_known_pareto, i_e_k, j_e_k, "d_e_k", d_e_k)
        plotDistance(best_known_pareto, pareto_estimated, i_k_e, j_k_e, "d_k_e", d_k_e)
        plotDistance(best_known_pareto, pareto_real, i_k_r, j_k_r, "d_k_r", d_k_r)
        plotDistance(pareto_real, best_known_pareto, i_r_k, j_r_k, "d_r_k", d_r_k)
        
        ## Plot pareto real
        plt.plot(pareto_real[:,0], pareto_real[:,1], 'r', label='Real Pareto')

        ## Plot ordered pareto estimated
        F = pareto_estimated[np.argsort(pareto_estimated[:,1])]
        plt.plot(F[:,0], F[:,1], 'b', label='Estimated Pareto')


        ## Plot best known pareto
        best_known_pareto = get_pareto_undominated_by(self.Y)
        plt.plot(best_known_pareto[:,0], best_known_pareto[:,1], 'xg', markersize=10, label="Best Known Pareto")
        plt.legend(bbox_to_anchor=(1.4, 0.4))
        
        ## Compute paretos diameters
        di_r = diameter(pareto_real)
        di_e = diameter(pareto_estimated)
        di_k = diameter(best_known_pareto)
        
        ## Compute mean distances
        dm_e_r = average_directed_haussdorf_distance(pareto_estimated, pareto_real)
        dm_r_e = average_directed_haussdorf_distance(pareto_real, pareto_estimated)
        dm_r_k = average_directed_haussdorf_distance(pareto_real, best_known_pareto)
        dm_k_r = average_directed_haussdorf_distance(best_known_pareto, pareto_real)
        dm_e_k = average_directed_haussdorf_distance(pareto_estimated, best_known_pareto)
        dm_k_e = average_directed_haussdorf_distance(best_known_pareto, pareto_estimated)
        
        metrics = {
            'di_r'  : di_r,
            'di_e'  : di_e,
            'di_k'  : di_k,
            'dm_e_r' : dm_e_r,
            'dm_r_e' : dm_r_e,
            'dm_e_k' : dm_e_k,
            'dm_k_e' : dm_k_e,
            'dm_k_r' : dm_k_r,
            'dm_r_k' : dm_r_k,
            'd_e_r' : d_e_r,
            'd_r_e' : d_r_e,
            'd_e_k' : d_e_k,
            'd_k_e' : d_k_e,
            'd_k_r' : d_k_r,
            'd_r_k' : d_r_k,
            'hp_e'  : getHyperVolume(pareto_estimated[np.argsort(pareto_estimated[:,0])]),
            'hp_r'  : getHyperVolume(pareto_real[np.argsort(pareto_real[:,0])]),
            'hp_k'  : getHyperVolume(best_known_pareto[np.argsort(best_known_pareto[:,0])]),
            'd_current_previous' : d_current_previous,
            'xd_current_previous' : xd_current_previous
        }

        if saveparetos:  
            plt.savefig("ImagesExp/"+str(len(self.X))+'.png')
        if showparetos:
            plt.show()
        plt.clf()
        return metrics

    def evaluateNoRealPareto(self, showparetos: bool = False, saveparetos: bool = False):
        ## Computation of Pareto Estimated
        problem = GPProblem(self)
        res = minimize(problem,
                NSGA2(),
                get_termination("n_gen", 40),
                save_history=True,
                verbose=False)
        pareto_estimated = res.F

        ## Computation of best known pareto
        best_known_pareto = get_pareto_undominated_by(self.Y)

        (d_e_k, _, _) = directed_hausdorff(pareto_estimated, best_known_pareto)
        (d_k_e, _, _) = directed_hausdorff(best_known_pareto, pareto_estimated)

        dm_e_k = average_directed_haussdorf_distance(pareto_estimated, best_known_pareto)
        dm_k_e = average_directed_haussdorf_distance(best_known_pareto, pareto_estimated)

        ## Plot ordered pareto estimated
        F = pareto_estimated[np.argsort(pareto_estimated[:,1])]
        plt.plot(F[:,0], F[:,1], 'b', label='Estimated Pareto')


        ## Plot best known pareto
        best_known_pareto = get_pareto_undominated_by(self.Y)
        plt.plot(best_known_pareto[:,0], best_known_pareto[:,1], 'xg', markersize=10, label="Best Known Pareto")
        plt.legend(bbox_to_anchor=(1.4, 0.4))
        if showparetos:
            plt.show()
        if saveparetos:  
            plt.savefig("ImagesExp/"+str(len(self.X))+'.png')
        plt.clf()
        return d_e_k+d_k_e, dm_e_k+dm_k_e

class MultiGPR(object):
    def __init__(self, X = None, Y = None, noise_variance=0.01):
        self.GPRs = [
            gpflow.models.GPR(
                [X, Y[:,i:i+1]],
                kernel = gpflow.kernels.SquaredExponential(), 
                mean_function = gpflow.mean_functions.Constant(),
                noise_variance = noise_variance
            )
            for i in range(Y.shape[-1]) 
        ]
        self.opt = gpflow.optimizers.Scipy()

    def optimizeKernel(self):
        for GPR in self.GPRs:
            self.opt.minimize(
                GPR.training_loss, 
                variables=GPR.trainable_variables)

    def printGPRs(self):
        for GPR in self.GPRs:
            print_summary(GPR)

    def predict_y(self, xx):

        mean_vars = tf.concat([GPR.predict_y(xx) for GPR in self.GPRs], axis=-1)
        mean = mean_vars[0]
        var = mean_vars[1]
        return mean, var

    def predict_f_samples(self, xx, n_samples):
        presamples = [GPR.predict_f_samples(xx, n_samples) for GPR in self.GPRs]
        samples = tf.concat(presamples[:], axis=-1)
        return samples        

