import tensorflow as tf 
import yaml 
import numpy as np 
import replay 
import globals 

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

upper_bound = config['upper_bound']
lower_bound = config['lower_bound']


def policy(state, noise_object):
    sampled_actions = tf.squeeze(globals.actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]
