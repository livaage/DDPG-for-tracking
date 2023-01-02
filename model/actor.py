import tensorflow as tf
from tensorflow.keras import layers
import yaml
#from base.model_base import BaseModel


#come back and parameterise with https://github.com/kuangliu/pytorch-cifar/blob/master/models/regnet.py 

#change config 
with open("/home/lhv14/DDPG/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


num_states = config['num_states']
num_actions = config['num_actions']
#
upper_bound = config['upper_bound']
#upper_bound = config['upper_bound']
lower_bound = config['lower_bound']


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(512, activation="relu")(inputs)
    out = layers.Dense(64, activation="relu")(out)


    outputs = layers.Dense(2, activation="tanh", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound

    model = tf.keras.Model(inputs, outputs)
    return model
