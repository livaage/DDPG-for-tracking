import tensorflow as tf
from tensorflow.keras import layers
import yaml

#fix 
with open("/home/lhv14/GCRL/DDPG/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


num_states = config['num_states']
num_actions = config['num_actions']
num_close_hits = config['num_close_hits']
num_comp_hits = config['num_comp_hits']
#
upper_bound = config['upper_bound']
#upper_bound = config['upper_bound']
lower_bound = config['lower_bound']



def get_critic():
    # State as input
    first_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(512, activation="relu")(state_input) 
    #state_out = layers.Dense(1024, activation="relu")(state_out)
    state_out = layers.Dense(256, activation="relu")(state_out)

    state_out = layers.Dense(64, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_close_hits, num_actions))
    
    action_out = layers.Dense(256, activation="relu")(action_input) 
    #action_out = layers.Dense(1024, activation="relu")(action_out)

    #action_out = layers.Dense(512, activation="relu")(action_out)
    #action_out = layers.Dense(64, activation="relu")(action_out)

    action_out = layers.Dense(64, activation="relu")(action_out)


    #comp_out = layers.Dense(32, activation="relu")(action_input) 

    action_convolution = layers.Conv1D(256, kernel_size=num_close_hits, activation='relu')(action_input)
    #action_convolution = layers.MaxPooling1D((4))(action_convolution)

    action_convolution = layers.Flatten()(action_convolution)
    #action_convolution = layers.Dense(512, activation="relu")(action_convolution) 

    #concat = layers.Concatenate()([state_out, action_convolution])
    #concat = layers.Concatenate()([state_out, action_out])
    concat = tf.keras.backend.repeat(state_out, n=num_close_hits)
    concat2 = layers.Concatenate()([concat, action_out])

    out = layers.Dense(512, activation="relu")(concat2)
    out = layers.Dense(128, activation="relu")(out)

    out = layers.Dense(32, activation="relu")(out)
    outputs = layers.Dense(1, activation="sigmoid")(out)
#kernel_initializer=first_init

    model = tf.keras.Model([state_input, action_input], outputs)
    return model
