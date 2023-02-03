import numpy as np 

def calc_distance(m, b, point_z, point_r): 
    d = np.abs(m*point_z - point_r +b )/np.sqrt(m**2+1)
    return d

def find_m_b(hit1, hit2):
    if  hit2.z != hit1.z: 
        m = (hit2.r - hit1.r)/(hit2.z - hit1.z)
    else: 
         m = 100
    b = hit2.r - m*hit2.z
    return m,b 


def find_m_b_no_df(hit1, hit2):
    if  hit2[0] != hit1[0]: 
        m = (hit2[1] - hit1[1])/(hit2[0] - hit1[0])
    else: 
         m = 100
    b = hit2[1] - m*hit2[0]
    return m,b 

def find_n_closest_hits(point_z, point_r, hits, n): 
    d = np.sqrt((point_z - hits.z)**2 + (point_r - hits.r)**2)
    closest_n_hits = hits.iloc[np.argsort(d)[:n]] 
    return closest_n_hits