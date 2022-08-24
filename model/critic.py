import tensorflow as tf
from tensorflow.keras import layers
import yaml


#fix 
with open("/home/lhv14/DDPG/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


num_states = config['num_states']
num_actions = config['num_actions']
#
upper_bound = config['upper_bound']
#upper_bound = config['upper_bound']
lower_bound = config['lower_bound']



def get_critic():
    # State as input
    first_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(32, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(64, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(32, activation="relu")(concat)
    out = layers.Dense(32, activation="relu")(out)
   # out = layers.Rescaling(1/1000)(out)
    outputs = layers.Dense(2,kernel_initializer=first_init)(out)
    
    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model
