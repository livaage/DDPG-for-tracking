import numpy as np 
#from utils.find_compatible_hits import Find_Compatible_Hits_ModuleMap_Line 
#from utils.find_compatible_hits_dev import Find_Compatible_Hits_ModuleMap_Line_New
import tensorflow as tf 
import csv 
import pandas as pd
from utils.geometry import find_n_closest_hits  
import yaml

with open("/home/lhv14/GCRL/DDPG/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

num_close_hits = config['num_close_hits']

#f = open("evaluation/comp_hits_qualities.csv", "w")
#writer = csv.writer(f)
#writer.writerow(["cor_z", "cor_r", "comp_hit_z", "comp_hit_r", "quality", "rank", "is right", "is chosen"])

def policy_maxQ(state, critic, comp_hits_z_r):


    state = np.tile(state, (num_close_hits, 1)) 

    #repeated 
    comp_all = np.tile(comp_hits_z_r, (num_close_hits,1)).reshape((-1, num_close_hits, 2))
    

    model_input = [tf.convert_to_tensor(state), tf.convert_to_tensor(comp_hits_z_r), tf.convert_to_tensor(comp_all)]
    #print("model input", model_input)
    q_vals = critic(model_input)    
    #print(q_vals)
    
    q_vals = np.array(q_vals).flatten()

    best_q_ix = np.argmax(q_vals) 
    # little step to make sure it isn't just learning to always have the same quality and chose the first ordered hit 
    best_qs = np.argwhere(q_vals == np.amax(q_vals)).flatten()
    # if len(best_qs) > 1: 
    #     best_q_ix = np.random.choice(best_qs)

    best_action = comp_hits_z_r[best_q_ix] 
    return_best_action = np.max(q_vals)

    return best_action, return_best_action, q_vals



