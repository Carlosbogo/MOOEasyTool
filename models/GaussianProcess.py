"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""

import numpy as np
import matplotlib.pyplot as plt

import gpflow

class GaussianProcess(object):
    def __init__(self, O:int, C:int, d:int, lowerBound: float, upperBound: float, kernel, X = None, Y = None, noise_variance=0.01):
        self.O = O
        self.C = C
        self.d = d
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.kernel = kernel
        self.X = X
        self.Y = Y
        self.noise_variance = noise_variance
        self.opt = gpflow.optimizers.Scipy()
        self.GPR = None

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
            kernel= self.kernel, 
            noise_variance=self.noise_variance)

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
        xx = np.linspace(self.lowerBound, self.upperBound, 100).reshape(100, 1)

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
        xx = np.linspace(self.lowerBound, self.upperBound, 100).reshape(100, 1)

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

    def plotMESMO(self, x_best, acq, x_tries, acqs):
        fig, axs = plt.subplots(nrows = self.O+self.C+1, ncols=self.d)
        xx = np.linspace(self.lowerBound, self.upperBound, 100).reshape(100, 1)

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
                    axs[i, j].axvline(x=x_best[j], color='r')
                
                for i in range(self.O, self.O+self.C):
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
                        axs[i, j].set_ylabel("c"+str(i))
                    axs[i, j].axvline(x=x_best[j], color='r')
                axs[self.O+self.C, j].plot(x_tries[:,j],acqs,'o', markersize=1)
                axs[self.O+self.C, j].plot([x_best[j]], [acq],'or', markersize=4)
                axs[self.O+self.C, j].set_ylim(acq-0.2, acq+2.2)

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

    def plotACQS(self, x_tries, acqs,x_best, acq):
        fig, axs = plt.subplots(nrows = self.O+1, ncols=self.d)
        xx = np.linspace(self.lowerBound, self.upperBound, 100).reshape(100, 1)

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
