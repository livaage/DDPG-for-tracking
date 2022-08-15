import circle_fit as cf
import numpy as np 

def estimate_momentum(data): 
    xc,yc,r,_ = cf.least_squares_circle((data))
    pt = r*0.01*0.3*3.8  

    return pt 

def dip_angle(dr, dz): 
    return np.tan(dr/dz)

def azimuthal_angle(dx, dy): 
    #print(dx, dy)
    #x = np.tan(dy, dx)
    return np.tan(dy/dx)