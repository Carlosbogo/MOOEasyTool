import numpy as np

def gramacy(x):
  
  term1 = np.sin(10*np.pi*x) / (2*x)
  term2 = (x-1)**4
  
  return term1 + term2
