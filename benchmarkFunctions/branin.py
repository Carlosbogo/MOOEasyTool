import numpy as np

def branin(xx, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):

    x1 = xx[0]
    x2 = xx[1]

    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s * (1-t)*np.cos(x1)

    return term1 + term2 + s