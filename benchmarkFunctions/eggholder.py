import numpy as np

def eggholder(xx):

  if len(xx)!=2:
    ValueError('Bukin6 function must have dimension 2, dimension',len(xx),'was found.')

  x1 = xx[0]
  x2 = xx[1]

  term1 = -(x2+47) * np.sin(np.sqrt(abs(x2+x1/2+47)))
  term2 = -x1 * np.sin(np.sqrt(abs(x1-(x2+47))))

  return term1 + term2

