import numpy as np

def rosenbrock(xx):
  
  # xi = xx[:-1]
  # xnext = xx[1:]
	
  # sum = np.sum(100*((xnext-xi**2)**2) + (xi-1)**2)
	
  x1 = xx[0]
  x2 = xx[1]

  term1 = 100*((x2 - x1**2)**2)
  term2 = (x1-1)**2

  return term1+term2
