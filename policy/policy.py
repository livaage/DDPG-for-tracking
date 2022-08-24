import pdb
import tensorflow as tf 
import yaml 
import numpy as np 
import replay 
import pandas as pd 
import json 
from utils.find_compatible_hits import Find_Compatible_Hits_ModuleMap_Line




pd.set_option("precision", 16)

with open("/home/lhv14/DDPG/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

upper_bound = config['upper_bound']
lower_bound = config['lower_bound']


def policy(state, noise_object1, noise_object2, actor_model, comp):
    sampled_actions = tf.squeeze(actor_model(state))
    noise1 = noise_object1()
    noise2 = noise_object2()
    # Adding noise to action
    #sampled_actions1 = sampled_actions.numpy() + noise1 
    sampled_actions1 = sampled_actions.numpy()[0] + noise1 
    sampled_actions2 = sampled_actions.numpy()[1] + noise2
    sampled_actions = np.array([sampled_actions1, sampled_actions2]).flatten()

    # We make sure action is within the boundaries of the environment 
    clipped_action = np.clip(sampled_actions, lower_bound, upper_bound)
    print("legal action", clipped_action)
    # now make sure it is one of the allowed hits
    #new_action = find_closest_allowed_next_states(state, legal_action)
    #print(state[0][2:].numpy(), state[0][:2].numpy())
    close_hits = comp.get_comp_hits(state[0][2:].numpy(), state[0][:2].numpy(), 10)
    #print(close_hits)
    #new_action = close_hits.iloc[0][['z', 'r']].values  
    new_action = find_closest_legal_action(close_hits, clipped_action)
    # printing both to see that the action before correction is very off and does not learn 
    #print("new action", new_action)
    return [np.squeeze(new_action)]

def find_closest_legal_action(close_hits, clipped_action): 
    distance = np.sqrt((close_hits['z'] - clipped_action[0])**2 + (close_hits['r'] - clipped_action[1])**2)
    ix = np.argmin(distance) 
    closest_hit = close_hits.iloc[ix][['z', 'r']] 
    if len(closest_hit) == 0: 
        closest_hit = clipped_action
    return closest_hit 



