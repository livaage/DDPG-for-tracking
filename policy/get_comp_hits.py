import numpy as np 
#from utils.find_compatible_hits import Find_Compatible_Hits_ModuleMap_Line 
#from utils.find_compatible_hits_dev import Find_Compatible_Hits_ModuleMap_Line_New
import tensorflow as tf  
import pandas as pd
from utils.geometry import find_n_closest_hits  
import yaml


with open("/home/lhv14/GCRL/DDPG/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

num_close_hits = config['num_close_hits']


#@numba.jit
def get_comp_hits(state, comp, correct_hit_id, dope=False):

    """takes the current state and the actual correct hit for the next state """ 

    hit2_df = comp.hit_df(state[:2])

    hit_number = hit2_df.hit_number 

    # prev_two_hits = comp.hits.iloc[int(max([hit_number-2, 0])):int(hit_number)]
    # print(prev_two_hits)

    # if len(prev_two_hits) < 2: 
    #     prev_two_hits = prev_two_hits.append(prev_two_hits.iloc[0])

    comp_hits, done = comp.get_comp_hits(hit2_df, state[2], state[3], num_close_hits) 
    #comp_hits, done = comp.get_comp_hits(prev_hit, hit2_df, num_close_hits) 
    # need the format to be consistent 
    if len(comp_hits) < num_close_hits: 
        try: 
            added_rows = pd.DataFrame([comp_hits.iloc[0]]*(num_close_hits-len(comp_hits))) 
        except: 
            print("the comp hits is ", comp_hits)
        comp_hits = pd.concat([comp_hits, added_rows])
    comp_hits = comp_hits.reset_index() 

    #print(comp_hits[['z', 'r', 'x', 'y']])
    cor = comp.get_hit(correct_hit_id).squeeze()
    #print("cor in get comp is ", cor[['z', 'r']])
    #if including, remmeber to change from max to median for the next hit reward
    if (dope) and (int(cor.hit_id) not in comp_hits.hit_id.values): 
        #print(int(cor.hit_id), comp_hits.hit_id)
        comp_hits.loc[comp_hits.index[num_close_hits-1]] = cor
    #print("comp hits in comp is", comp_hits[['z', 'r']])
    #cor instead 
    #random_choice = np.random.choice(len(comp_hits))
    # comp_hits = comp_hits.iloc[random_choice]
    # comp_hits = pd.concat([comp_hits, comp_hits])
    #print(comp_hits)
    

    #change? 
    #comp_hits = comp_hits.sort_values(['r'])
    #print("")
    
    comp_hits = comp_hits.sample(frac=1)
    comp_hits_z_r_x_y = comp_hits[['z', 'r', 'x', 'y']].values
    #print("in comp: \n", "the state:", state, "the comps ", comp_hits_z_r)



    #rewards = comp.get_rank_reward(comp_hits, cor)
    #rewards = [comp.get_reward(comp_hits.iloc[i], cor) for i in range(len(comp_hits))]

    rewards = comp.get_reward_binary(comp_hits, cor)
    #print("hit2 is", hit2_df[['z', 'r']], "and the compy bois are ", comp_hits_z_r, "correct is", cor[['z', 'r']])
    return comp_hits_z_r_x_y, rewards, cor.hit_id


