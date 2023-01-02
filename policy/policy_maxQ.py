import numpy as np 
#from utils.find_compatible_hits import Find_Compatible_Hits_ModuleMap_Line 
#from utils.find_compatible_hits_dev import Find_Compatible_Hits_ModuleMap_Line_New
import tensorflow as tf 
import csv 
import pandas as pd
from utils.geometry import find_n_closest_hits  


f = open("evaluation/comp_hits_qualities.csv", "w")
writer = csv.writer(f)
writer.writerow(["hit2_z", "hit2_r", "comp_hit_z", "comp_hit_r", "quality"])

def policy_maxQ(state, critic, comp):

    #comp_hits, done = comp.get_comp_hits(state[2:], state[:2], 5)
    hit2_df = comp.hit_df(state[:2])
    hits = comp.get_all_hits()
    p = hits[hits['particle_id']==hit2_df.particle_id]
    p = p.sort_values(['r', 'z']).reset_index() 
    p_without_same_layer = p[p['unique_layer_id']!=hit2_df.unique_layer_id]
    try: 
        correct_hit = p_without_same_layer[p_without_same_layer['r'] > hit2_df.r].iloc[0]
    except: 
        correct_hit = hit2_df 
    

    if hit2_df.empty: 
        print("yes, it's empty", comp.hits, state[:2])
    comp_hits, done = comp.get_comp_hits(hit2_df, state[2], state[3], 10) 
    #comp_hits = pd.concat([comp_hits, pd.DataFrame([correct_hit])], axis=0)
    #also make sure the actual hits are in there 
    
    #comp_hits = comp_hits.append(comp.hits[comp.hits['particle_id']==-20750.0])
    #print("at this point comp is ", comp_hits)
    if len(comp_hits) < 10: 
        #print(len(comp_hits))
        added_rows = pd.DataFrame([comp_hits.iloc[0]]*(10-len(comp_hits))) 
        comp_hits = pd.concat([comp_hits, added_rows])
        #print(len(comp_hits), len(added_rows))
        #comp_hits_z_r = pd.concat([np.repeat(comp_hits.iloc[0], 10-(len(comp_hits_z_r))))
    
    comp_hits_z_r = comp_hits[['z', 'r']].values

    states= np.array([state] *10)
    close_hits = find_n_closest_hits(state[0], state[1], hits,1)
    #print(close_hits[['z', 'r']].values)
    #print(close_hits)
    #state_actions = [tf.convert_to_tensor(states), tf.convert_to_tensor(comp_hits_z_r), tf.convert_to_tensor(close_hits[['z', 'r']].values)] 
    #print("state actions", states, "comp hits", comp_hits_z_r)
    #print("state actions", len(states), len(close_hits), len(comp_hits_z_r))
    #q_vals = critic(state_actions) 
    

    # this bit is changing 
    # q_vals = []
    # for i in range(len(comp_hits)): 
    #     comp_hit = comp_hits_z_r[i, ]
    #     #print(comp)
    #     state_actions = [np.array([tf.convert_to_tensor(state)]), np.array([tf.convert_to_tensor(comp_hits_z_r)]), np.array([tf.convert_to_tensor(close_hits[['z', 'r']].values)]), np.array([tf.convert_to_tensor(comp_hit)])]
    #     q_val = critic(state_actions, training=True)
    #     q_vals.append(q_val)

    state_actions = [np.array([tf.convert_to_tensor(state)]), np.array([tf.convert_to_tensor(comp_hits_z_r)])]
    q_vals = critic(state_actions)
    #state_actions = [np.array([tf.convert_to_tensor(state)]), np.array([tf.convert_to_tensor(comp_hits_z_r)]), np.array([tf.convert_to_tensor(close_hits[['z', 'r']].values)])]
    #q_vals = critic(state_actions, training=True)
    q_vals = np.array(q_vals).flatten()
    #print("q vals", q_vals)

    best_q_ix = np.argmax(q_vals) 
    best_action = comp_hits_z_r[best_q_ix] 
    return_best_action = np.max(q_vals)

    #print(q_vals)
    #print("actions", comp_hits_z_r, "qualities", q_vals, "best_q_pos", best_q_ix, "best action", best_action)

    # for i in range(len(comp_hits)):
    #     comp_row = comp_hits.iloc[i]
    #     row = pd.DataFrame({
    #     'hit2_z': [hit2_df.z],
    #     'hit2_r': [hit2_df.r],  
    #     'comp_hit_z': [comp_row.z], 
    #     'comp_hit_r': [comp_row.r], 
    #     'quality': q_vals[i]})
    #     row.to_csv(f, mode='a', header=None, index=None)
        

#    print(best_action)

    return best_action, return_best_action, state_actions, q_vals, done  



