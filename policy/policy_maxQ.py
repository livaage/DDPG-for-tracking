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

f = open("evaluation/comp_hits_qualities.csv", "w")
writer = csv.writer(f)
writer.writerow(["cor_z", "cor_r", "comp_hit_z", "comp_hit_r", "quality", "rank", "is right", "is chosen"])

def policy_maxQ(state, critic, comp, correct_hit_id):

    """takes the current state and the actual correct hit for the next state """ 

    hit2_df = comp.hit_df(state[:2])


    comp_hits, done = comp.get_comp_hits(hit2_df, state[2], state[3], num_close_hits) 
    # need the format to be consistent 
    if len(comp_hits) < num_close_hits: 
        try: 
            added_rows = pd.DataFrame([comp_hits.iloc[0]]*(num_close_hits-len(comp_hits))) 
        except: 
            print("the comp hits is ", comp_hits)
        comp_hits = pd.concat([comp_hits, added_rows])
    comp_hits = comp_hits.reset_index() 

    particle = comp.get_particle(hit2_df.particle_id)

    cor = comp.get_hit(correct_hit_id).squeeze()
    #print("inside policy, the state is ", hit2_df[['z', 'r', 'unique_layer_id', 'hit_id']], "and the correct hit si ", cor[['z', 'r', 'unique_layer_id', 'hit_id']])
    # try: 
    #     cor = particle[particle['r'] > hit2_df.r].iloc[0].squeeze() 
    # except: 
    #     cor = particle.iloc[-1]
    #     done = True 
    #print("in policy the particle is", particle[['']], " hit2 df is ", state[:2], "the cor is", cor)
    #print("particle is ", particle[['unique_layer_id', 'hit_id', 'z', 'r']], " hit 2 is ", state[:2], "and cor is ", cor[['z', 'r', 'unique_layer_id']], "cor hit id", correct_hit_id)
    #print(cor)

    #print(comp_hits, cor)
    #try: 
    
    
    comp_hits.loc[comp_hits.index[4]] = cor
    
    
    #     print(comp_hits, cor)
    # except: 
    #     print(comp_hits, cor)
    #print(comp_hits)
    #print(comp_hits)
    # try: cor
    #     comp_hits.loc[comp_hits.index[1]] = cor
    # except: 
    #    #print("the comp hits thing", comp_hits, cor)
    #     pass


        #print(comp_hits, cor)
        #print("couldn't append it")
    # if cor.hit_id in comp_hits.hit_id.values: 
    #     print("it's here")
    # else: 
    #     print("it's not here")
    #print(comp_hits)
    comp_hits = comp_hits.sort_values(['r', 'z'])
    comp_hits_z_r = comp_hits[['z', 'r']].values

    rewards = [comp.get_reward(comp_hits.iloc[i], cor) for i in range(len(comp_hits))]
   
    state_actions = [np.array([tf.convert_to_tensor(state)]), np.array([tf.convert_to_tensor(comp_hits_z_r)])]
    q_vals = []
    for i in range(len(comp_hits_z_r)): 
        q = critic( [np.array([tf.convert_to_tensor(state)]), np.array([tf.convert_to_tensor(comp_hits_z_r[i])])])
        q_vals.append(q)
    
    
    #q_vals = critic(state_actions)
    q_vals = np.array(q_vals).flatten()

    best_q_ix = np.argmax(q_vals) 
    # little step to make sure it isn't just learning to always have the same quality and chose the first ordered hit 
    best_qs = np.argwhere(q_vals == np.amax(q_vals)).flatten()
    #print(q_vals, best_qs)
    if len(best_qs) > 1: 
        best_q_ix = np.random.choice(best_qs)

    #print(best_q_ix)

    best_action = comp_hits_z_r[best_q_ix] 
    best_action_df = comp_hits.iloc[best_q_ix]
    #print("best, df, comp", best_action, best_action_df, comp.hit_df(best_action))
    #print(best_action_df[['z', 'r', 'hit_id']], cor[['z', 'r', 'hit_id']], best_action_df.hit_id==cor.hit_id)
    return_best_action = np.max(q_vals)
    sorted_qvals = np.sort(q_vals)
    for i in range(len(comp_hits)):
        #print(np.where(sorted_qvals == q_vals[i])[0])
        is_chosen = (q_vals[i] == return_best_action)
        is_right = (comp_hits.iloc[i].hit_id == cor.hit_id)
        comp_row = comp_hits.iloc[i]
        # if ((is_right) & (is_chosen)): 
        #     print(best_action,  cor[['z', 'r']])
        #print(is_right, best_action, cor[['z', 'r']])
        row = pd.DataFrame({
        'cor_z': [cor.z],
        'cor_r': [cor.r],  
        'comp_hit_z': [comp_row.z], 
        'comp_hit_r': [comp_row.r], 
        'quality': q_vals[i], 
        'rank': [np.where(sorted_qvals == q_vals[i])[0]], 
        'is right': [is_right], 
        'is chosen': [is_chosen]})
        row.to_csv(f, mode='a', header=None, index=None)
        

#    print(best_action)

    return best_action, return_best_action, state_actions, q_vals, rewards, done  



