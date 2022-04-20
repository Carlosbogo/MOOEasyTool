import numpy as np

def directed_hausdorff(xx, yy):
    if len(xx)==0 or len(yy)==0:
        return float("Nan"), -1, -1
    zz = np.zeros((xx.shape[0], yy.shape[0]))

    for i_x, x in enumerate(xx):
        for i_y, y in enumerate(yy):
            zz[i_x, i_y]=np.linalg.norm(x-y)

    z_1 = np.amin(zz, axis=1)
    idx_y = np.argmin(zz, axis=1)
    
    z_2 = np.amax(z_1)
    idx_x = np.argmax(z_1)

    return z_2, idx_x, idx_y[idx_x]

def hausdorff(xx,yy):
    (d1,x1,y1) = directed_hausdorff(xx, yy)
    (d2,x2,y2) = directed_hausdorff(yy, xx)
    
    if d1>d2:
        return (d1,x1,y1)
    else:
        return (d2,x2,y2)

def average_directed_haussdorf_distance(xx, yy, p=1):
    if len(xx)==0 or len(yy)==0:
        return float("Nan"), -1, -1
    zz = np.zeros((xx.shape[0], yy.shape[0]))

    for i_x, x in enumerate(xx):
        for i_y, y in enumerate(yy):
            zz[i_x, i_y]=np.linalg.norm(x-y)**p
            
    return np.mean(np.amin(zz, axis=1))**(1/p)
    
def mean_distance(xx,yy):
    if len(xx)==0 or len(yy)==0:
        return float("Nan")
    zz = np.zeros((xx.shape[0], yy.shape[0]))

    for i_x, x in enumerate(xx):
        for i_y, y in enumerate(yy):
            zz[i_x, i_y]=np.linalg.norm(x-y)
    return np.mean(zz)
    
def diameter(xx):
    if len(xx)==0:
        return float("Nan")
    zz = np.zeros((xx.shape[0], xx.shape[0]))

    for i_x, x in enumerate(xx):
        for i_y, y in enumerate(xx):
            zz[i_x, i_y]=np.linalg.norm(x-y)
    
    return np.amax(zz)  
    
    
    
def getHyperVolume(front, reference = None):
    if reference is None:
        reference = [0,0]

    hypervolume = 0

    for el in front:
        hypervolume+=(el[0]-reference[0])*(el[1]-reference[1])
        reference[1]=el[1]
    return hypervolume