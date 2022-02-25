import tensorflow as tf 
import yaml 
import numpy as np 
import replay 
import globals 

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

upper_bound = config['upper_bound']
lower_bound = config['lower_bound']


def policy(state, noise_object1, noise_object2):
    sampled_actions = tf.squeeze(globals.actor_model(state))
    noise1 = noise_object1()
    noise2 = noise_object2()
    # Adding noise to action
    #sampled_actions1 = sampled_actions.numpy() + noise1 
    sampled_actions1 = sampled_actions.numpy()[0] + noise1 
    sampled_actions2 = sampled_actions.numpy()[1] + noise2
    sampled_actions = np.array([sampled_actions1, sampled_actions2]).flatten()
    #print("sample2", sampled_actions)
    #print()
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]
