import numpy as np
import sobol_seq
import minepy
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import norm, spearmanr

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination, get_sampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem

from models.GaussianProcess import GaussianProcess

def SingleObjectiveAcqGrid(function, GP: GaussianProcess, N = 12):
    xx = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBound-GP.lowerBound)+GP.lowerBound
    zz = function(GP, xx)
    return xx[np.argmax(zz)], np.amax(zz)
    
def SingleObjectiveAcq(function, GP: GaussianProcess):
    problem = SingleObjectiveProblem(function, GP)
    algorithm = NSGA2()
    res = minimize(problem,
                   algorithm,
                   verbose=False)
    if len(res.X.shape)==1:
        return res.X, res.F
    return res.X[0], res.F[0]
    
class SingleObjectiveProblem(Problem):    
    def __init__(self,function, GP: GaussianProcess):
        super().__init__(n_var=GP.d, n_obj=1, n_constr=0, xl=np.array([GP.lowerBound]*GP.d), xu=np.array([GP.upperBound]*GP.d))
        self.function = function
        self.GP = GP

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.column_stack( [-self.function(self.GP, X)] )

def random(GP, X):
    return np.random.choice(X)
        
def pi(GP: GaussianProcess, X, e = 0):
    """
        Probability of improvement: 
            * Assumes Minimization
            * Select Maxium value
            probability image of x, f(x)===N(m,v) is better than the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
        Returns:
            pi : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    mean, var = GP.GPR.predict_y(X)
    optimum = tf.math.reduce_min(GP.Y, axis = 0)
    
    z = -(optimum - e - mean)/tf.math.sqrt(var)
    return 1-norm.cdf(z)

def ei(GP: GaussianProcess, X, e = 0):

    """
        Expected improvement: 
            * Assumes Minimization
            * Select Maxium value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
        Returns:
            ei : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    mean, var = GP.multiGPR.predict_y(X)
    optimum = tf.math.reduce_min(GP.Y, axis = 0)
    
    z = -(optimum - e - mean)/tf.math.sqrt(var)
    return (optimum - e -mean)*(1-norm.cdf(z))+tf.math.sqrt(var)*norm.pdf(z)

def ucb(GP: GaussianProcess, X, beta = 0.1, e = 0):
    """
        Upper Confidence Bound: 
            * Assumes Minimization
            * Select Maximum value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
            beta : Optimistic control hyperparameter
        Returns:
            ucb  : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    mean, var = GP.GPR.predict_y(X)
    return -(mean - beta*var)

def mes(GP: GaussianProcess, X, e = 0, N = 12, M = 10):
    """
        Max-Value Entropy Search: 
            * Assumes Minimization
            * Select Maximum value
            mutual information between y and y*
        Input: 
            GP  : Single Objective Gaussian Process that models funcition
            X   : Numpy array of shape (-1,1) of candidates to evaluate
            N  : Fixes lenght of samples to 2**N
            M  : Number of samples
        Returns:
            mes : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    mean, var = GP.GPR.predict_y(X)
    
    xx = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBound-GP.lowerBound)+GP.lowerBound
    samples = GP.GPR.predict_f_samples(xx,M)
    optimums = tf.math.reduce_min(samples, axis = 1)
    
    acq = 0
    for optimum in optimums:
        z = -(optimum+e-mean)/tf.math.sqrt(var)
        pdf, cdf = norm.pdf(z), norm.cdf(z)
        acq += -tf.math.log(cdf)+z*pdf/2/(cdf)
        
    return acq/len(optimums)

def simulated_mes_bins(GP: GaussianProcess, e = 0, N = 12, M = 5):
    """
        Max-value Entropy Search Simulated: 
            * Assumes Minimization
            * Select Maximum value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
            N  : Fixes lenght of samples to 2**N
            M  : Fixes number of samples to M**2
        Returns:
            simulated_mes  : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    xx = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBounds[0]-GP.lowerBounds[0])+GP.lowerBounds[0]
    samples = GP.multiGPR.predict_f_samples(xx,M**2)
    optimums = tf.math.reduce_min(samples, axis = 1)
    
    optimums_distances = np.array(
        [[abs(o1-o2) for o2 in optimums] for o1 in optimums]
    )
    
    total_mean_d = np.mean(optimums_distances)
    optimum_i = -1
    max_acq = -1
    for i in range(xx.shape[0]):
        yy = samples[:,i,:]
        ii = np.argsort(yy, axis=0)[:,0]
        
        temp_mean = 0
        
        for m in range(M):
            subset_optimus_distance = optimums_distances[ii[m*M:(m+1)*M]][:, ii[m*M:(m+1)*M]]
            temp_mean+=np.mean(subset_optimus_distance)/M
            
        temp_acq = total_mean_d - temp_mean
        if max_acq < temp_acq:
            max_acq = temp_acq
            optimum_i = i
    return np.array(xx[optimum_i]), max_acq

def simulated_mes_correlation(GP: GaussianProcess, e = 0, N = 12, M = 5):
    """
        Max-value Entropy Search Simulated: 
            * Assumes Minimization
            * Select Maximum value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
            N  : Fixes lenght of samples to 2**N
            M  : Fixes number of samples to M
        Returns:
            simulated_mes  : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    xx = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBound-GP.lowerBound)+GP.lowerBound
    samples = GP.GPR.predict_f_samples(xx,M)[:,:,0]
    optimums = tf.math.reduce_min(samples, axis = 1)
   
    def fn(yy):
        return tfp.stats.correlation(yy, y=optimums, event_axis=None)
    
    cors = tf.map_fn(fn, tf.transpose(samples, [1, 0]))
    
    return np.array(xx[tf.argmax(cors)]),  tf.reduce_max(cors)
   
def simulated_mes_spearmanr_correlation(GP: GaussianProcess, e = 0, N = 12, M = 5):
    """
        Max-value Entropy Search Simulated: 
            * Assumes Minimization
            * Select Maximum value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
            N  : Fixes lenght of samples to 2**N
            M  : Fixes number of samples to M
        Returns:
            simulated_mes  : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    xx = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBound-GP.lowerBound)+GP.lowerBound
    samples = GP.GPR.predict_f_samples(xx,M)[:,:,0]
    optimums = tf.math.reduce_min(samples, axis = 1)
   
    def fn(yy):
        return spearmanr(yy, optimums)[0]
    
    cors = tf.map_fn(fn, tf.transpose(samples, [1, 0]))
    return np.array(xx[tf.argmax(cors)]),  tf.reduce_max(cors)
    
def simulated_mes_mic_e(GP: GaussianProcess, e = 0, N = 12, M = 5):
    """
        Max-value Entropy Search Simulated: 
            * Assumes Minimization
            * Select Maximum value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
            N  : Fixes lenght of samples to 2**N
            M  : Fixes number of samples to M
        Returns:
            simulated_mes  : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    xx = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBound-GP.lowerBound)+GP.lowerBound
    samples = GP.GPR.predict_f_samples(xx,M)[:,:,0]
    optimums = tf.math.reduce_min(samples, axis = 1)
    
    mic_e = minepy.MINE(alpha=0.6, c=15, est="mic_e")
    
    def fn_e(yy):
        mic_e.compute_score(yy, optimums)
        return mic_e.mic()
    
    cors = tf.map_fn(fn_e, tf.transpose(samples, [1, 0]))
    #print(tf.reshape(tf.where(cors==tf.reduce_max(cors)), (-1,)).shape)

    return np.array(xx[tf.argmax(cors)]),  tf.reduce_max(cors)

def simulated_mes_mic_approx(GP, e = 0, N = 12, M = 5):
    """
        Max-value Entropy Search Simulated: 
            * Assumes Minimization
            * Select Maximum value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
            N  : Fixes lenght of samples to 2**N
            M  : Fixes number of samples to M
        Returns:
            simulated_mes  : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    xx = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBound-GP.lowerBound)+GP.lowerBound
    samples = GP.GPR.predict_f_samples(xx,M)[:,:,0]
    optimums = tf.math.reduce_min(samples, axis = 1)
    
    mic_approx = minepy.MINE(alpha=0.6, c=15, est="mic_approx")

    def fn_approx(yy):
        mic_approx.compute_score(yy, optimums)
        return mic_approx.mic()
    cors = tf.map_fn(fn_approx, tf.transpose(samples, [1, 0]))
    #print(tf.reshape(tf.where(cors==tf.reduce_max(cors)), (-1,)).shape)
    #print(tf.reshape(tf.where(tf.math.logical_and(cors2==tf.reduce_max(cors2), cors==tf.reduce_max(cors))), (-1,)).shape)
    
    return np.array(xx[tf.argmax(cors)]),  tf.reduce_max(cors)    

def simulated_mes_covariance(GP, e = 0, N = 12, M = 5):
    """
        Max-value Entropy Search Simulated: 
            * Assumes Minimization
            * Select Maximum value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
            N  : Fixes lenght of samples to 2**N
            M  : Fixes number of samples to M
        Returns:
            simulated_mes  : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    xx = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBound-GP.lowerBound)+GP.lowerBound
    samples = GP.GPR.predict_f_samples(xx,M)[:,:,0]
    optimums = tf.math.reduce_min(samples, axis = 1)
    
    def fn(yy):
        return tfp.stats.covariance(yy, y=optimums, event_axis=None)

    covs = tf.map_fn(fn, tf.transpose(samples, [1, 0]))
    
    return np.array(xx[tf.argmax(covs)]),  tf.reduce_max(covs)
    
def simulated_mes_distancecorrelation(GP, e = 0, N = 12, M = 5):
    """
        Max-value Entropy Search Simulated: 
            * Assumes Minimization
            * Select Maximum value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
            N  : Fixes lenght of samples to 2**N
            M  : Fixes number of samples to M
        Returns:
            simulated_mes  : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    xx = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBound-GP.lowerBound)+GP.lowerBound
    samples = GP.GPR.predict_f_samples(xx,M)[:,:,0]
    optimums = tf.math.reduce_min(samples, axis = 1)
    
    def getDistancesMatrix(A):
        A = tf.reshape(A, (-1,1))
        R = A*A
        distance_matrix = tf.math.sqrt( tf.maximum( R-2*tf.matmul(A, A, transpose_b=True)+tf.transpose(R), 1e-9 ) )
        return distance_matrix
    
    def getDoublyDistances(A):
        A_j = tf.reshape(tf.reduce_mean(A, axis=1), (-1,1))
        Aj_ = tf.reshape(tf.reduce_mean(A, axis=0), (1,-1))
        return A - A_j - Aj_ + tf.reduce_mean(A)
    
    optimums_distances = getDoublyDistances(getDistancesMatrix(optimums))
            
    def fn(yy):
        yy_distances = getDoublyDistances(getDistancesMatrix(yy))
        return tf.reduce_mean(optimums_distances*yy_distances)/  \
                tf.sqrt( tf.reduce_mean(yy_distances*yy_distances)*tf.reduce_mean(optimums_distances*optimums_distances) )
    dcors = tf.map_fn(fn, tf.transpose(samples, [1, 0]))
    
    if  tf.math.is_nan(tf.reduce_max(dcors)):
        import pdb
        pdb.set_trace()
    return np.array(xx[tf.argmax(dcors)]),  tf.reduce_max(dcors)
    
def simulated_mes_distancecovariance(GP, e = 0, N = 12, M = 5):
    """
        Max-value Entropy Search Simulated: 
            * Assumes Minimization
            * Select Maximum value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
            N  : Fixes lenght of samples to 2**N
            M  : Fixes number of samples to M
        Returns:
            simulated_mes  : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    xx = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBound-GP.lowerBound)+GP.lowerBound
    samples = GP.GPR.predict_f_samples(xx,M)[:,:,0]
    optimums = tf.math.reduce_min(samples, axis = 1)
    
    def getDistancesMatrix(A):
        A = tf.reshape(A, (-1,1))
        R = A*A
        distance_matrix = tf.math.sqrt( tf.maximum( R-2*tf.matmul(A, A, transpose_b=True)+tf.transpose(R), 1e-9 ) )
        return distance_matrix
    
    def getDoublyDistances(A):
        A_j = tf.reshape(tf.reduce_mean(A, axis=1), (-1,1))
        Aj_ = tf.reshape(tf.reduce_mean(A, axis=0), (1,-1))
        return A - A_j - Aj_ + tf.reduce_mean(A)
    
    optimums_distances = getDoublyDistances(getDistancesMatrix(optimums))
            
    def fn(yy):
        yy_distances = getDoublyDistances(getDistancesMatrix(yy))
        return tf.reduce_mean(optimums_distances*yy_distances)
    
    dcors = tf.map_fn(fn, tf.transpose(samples, [1, 0]))
    
    return np.array(xx[tf.argmax(dcors)]),  tf.reduce_max(dcors)


def simulated_mes_distances_correlation(GP, e = 0, N = 12, M = 5):
    """
        Max-value Entropy Search Simulated: 
            * Assumes Minimization
            * Select Maximum value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
            N  : Fixes lenght of samples to 2**N
            M  : Fixes number of samples to M
        Returns:
            simulated_mes  : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    xx       = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBound-GP.lowerBound)+GP.lowerBound
    samples  = GP.GPR.predict_f_samples(xx,M)[:,:,0]
    optimums = tf.math.reduce_min(samples, axis = 1)

    def getDistances(A):
        A = tf.reshape(A, (-1,1))
        R = A*A
        distance_matrix = R-2*tf.matmul(A, A, transpose_b=True)+tf.transpose(R)
        
        ones   = tf.ones_like(distance_matrix)
        mask_a = tf.linalg.band_part(ones, 0, -1)         # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(ones, 0, 0)          # Diagonal matrix of 0s and 1s
        mask   = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask

        return  tf.boolean_mask(distance_matrix, mask)
    
    optimums_distances = getDistances(optimums)
            
    def fn(yy):
        return tfp.stats.correlation(getDistances(yy), y=optimums_distances, event_axis=None)
    cors = tf.map_fn(fn, tf.transpose(samples, [1, 0]))
        
    return np.array(xx[tf.argmax(cors)]),  tf.reduce_max(cors)
    
def simulated_mes_distances_covariance(GP, e = 0, N = 12, M = 5):
    """
        Max-value Entropy Search Simulated: 
            * Assumes Minimization
            * Select Maximum value
            expected improvement of image of x, f(x)===N(m,v) compared to the actual optimum y*
        Input: 
            GP : Single Objective Gaussian Process that models funcition
            X  : Numpy array of shape (-1,1) of candidates to evaluate
            N  : Fixes lenght of samples to 2**N
            M  : Fixes number of samples to M
        Returns:
            simulated_mes  : Numpy array of shape (-1,1) with acquisition value for candidates
    """
    xx = sobol_seq.i4_sobol_generate(GP.d,2**N)*(GP.upperBound-GP.lowerBound)+GP.lowerBound
    samples = GP.GPR.predict_f_samples(xx,M)[:,:,0]
    optimums = tf.math.reduce_min(samples, axis = 1)

    def getDistances(A):
        A = tf.reshape(A, (-1,1))
        R = A*A
        distance_matrix = R-2*tf.matmul(A, A, transpose_b=True)+tf.transpose(R)
        
        ones = tf.ones_like(distance_matrix)
        mask_a = tf.linalg.band_part(ones, 0, -1)       # Upper triangular matrix of 0s and 1s
        mask_b = tf.linalg.band_part(ones, 0, 0)        # Diagonal matrix of 0s and 1s
        mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask

        return  tf.boolean_mask(distance_matrix, mask)
    
    optimums_distances = getDistances(optimums)
            
    def fn(yy):
        return tfp.stats.covariance(getDistances(yy), y=optimums_distances, event_axis=None)
    cors = tf.map_fn(fn, tf.transpose(samples, [1, 0]))
        
    return np.array(xx[tf.argmax(cors)]),  tf.reduce_max(cors)
