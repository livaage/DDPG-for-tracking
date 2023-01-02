import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd


md = pd.read_csv('/home/lhv14/new_md_hitbased.csv',  header=[0], index_col=[0, 1,2])
md = md.reset_index()
md.rename(columns = {'level_1':'volume_id', 'level_2':'layer_id'}, inplace = True)
hor_vol = [8, 13, 17]
ver_vol = [7, 9, 12, 14, 16, 18]



def plot_particle(particle_df): 
    plt.plot(particle_df.z, particle_df.r, "kx") 
    plt.show() 

def plot_particle_layers(particle_df): 
    plt.plot(particle_df.z, particle_df.r, "kx") 
    plt.plot(p.z, p.r, "kx")
    for vol in hor_vol:
        v = md[md['volume_id']==vol]
        for i in range(v.shape[0]):
            row = v.iloc[i]
            plt.plot([row.z_min,row.z_max], [row.r_mean, row.r_mean], "r")

    for vol in ver_vol:
        v = md[md['volume_id']==vol]
        for i in range(v.shape[0]):
            row = v.iloc[i]
            plt.plot([row.z_mean,row.z_mean], [row.r_min, row.r_max], "r")
    plt.show()
