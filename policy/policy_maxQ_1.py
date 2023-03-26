import numpy as np 
#from utils.find_compatible_hits import Find_Compatible_Hits_ModuleMap_Line 
#from utils.find_compatible_hits_dev import Find_Compatible_Hits_ModuleMap_Line_New
import tensorflow as tf 
import csv 
import pandas as pd
from utils.geometry import find_n_closest_hits  
import yaml

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

num_close_hits = config['num_close_hits']

#f = open("evaluation/comp_hits_qualities.csv", "w")
#writer = csv.writer(f)
#writer.writerow(["cor_z", "cor_r", "comp_hit_z", "comp_hit_r", "quality", "rank", "is right", "is chosen"])

def policy_maxQ(state, critic, comp_hits_z_r_x_y):



    state = np.tile(state, (num_close_hits, 1)) 
    #repeated 
    comp_all = np.tile(comp_hits_z_r_x_y, (num_close_hits,1)).reshape((-1, num_close_hits, 4))
    


    
    model_input = [tf.convert_to_tensor(state), tf.convert_to_tensor(comp_hits_z_r_x_y), tf.convert_to_tensor(comp_all)]
    q_vals = []
    #for i in comp_hits_z_r: 
       # model_input = [tf.convert_to_tensor(state), tf.convert_to_tensor(i)]

    #model_input = [tf.expand_dims(state, axis=1), tf.expand_dims(i, axis=1)]
        #print("policy input", model_input)
    #print("model input", model_input)
        #print("calling in loop")
        #print([state, i])
        # tf.reshape(input_tensor,shape=(1,n))    # n is the number of samples, feature tensor have 
        #print("reshaped", np.array(state).reshape(1,4))
        #q_vals.append(critic([np.array(state[:2]).reshape(1,2), np.array(i).reshape(1, 2), comp_hits_z_r.reshape(1, num_close_hits, 2)])) 
    #print(q_vals)from keras.utils import to_categorical 
    q_vals = critic(model_input)    

    #print(q_vals)
    
    q_vals = np.array(q_vals).flatten()
    best_q_ix = np.argmax(q_vals) 
    #np.argmax(q_vals, axis=1)
    #best_best = np.argmin(best_q_ix)
    #print(q_vals, best_q_ix)
    # little step to make sure it isn't just learning to always have the same quality and chose the first ordered hit 
    best_qs = np.argwhere(q_vals == np.amax(q_vals)).flatten()
    # dice = np.random.choice(range(100))
    # if dice > 70: 
    #     best_q_ix = np.random.choice(len(comp_hits_z_r))
    #     print("random")

    # # pick the worst if it's saturating to discourage saturation 
    if len(set(q_vals))==1: 
        print("were here")
        best_q_ix = len(comp_hits_z_r_x_y)-2
        #print("saturation going on")
    # if len(best_qs) > 1: 
    #     best_q_ix = np.random.choice(best_qs)

    #random_choice = np.random.choice(range(len(comp_hits_z_r)))

    best_action = comp_hits_z_r_x_y[best_q_ix] 
    #return_best_action = np.max(q_vals)
    return_best_action = q_vals[best_q_ix]
    

    return best_action, return_best_action, q_vals



