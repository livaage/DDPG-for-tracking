import tensorflow as tf
from tensorflow.keras import layers
import yaml

#fix 
with open("/home/lhv14/GCRL/DDPG/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


num_states = config['num_states']
num_actions = config['num_actions']
num_neighbours = config['num_neighbours']
num_comp_hits = config['num_comp_hits']
#
upper_bound = config['upper_bound']
#upper_bound = config['upper_bound']
lower_bound = config['lower_bound']



def get_critic():
    # State as input
    first_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(128, activation="relu")(state_input) 
    state_out = layers.Dense(32, activation="relu")(state_out)
    
    #mha = layers.MultiHeadAttention(num_heads=10, key_dim=10)
    #mha = layers.Attention()
    # Close hits as input 
    #close_input = layers.Input(shape=(1, num_actions))
    #close_input_attention =  mha(state_iun, close_input)
    # close_out = layers.Dense(64, activation="relu")(close_input)
    # close_out = layers.Dense(32, activation="relu")(close_out)
    #attention_1 = layers.Attention()([state_out, close_out])
    #print("att")

    # Action as input
    action_input = layers.Input(shape=(10, num_actions))
    
    action_out = layers.Dense(2048, activation="relu")(action_input) 
    action_out = layers.Dense(512, activation="relu")(action_out)

    action_out = layers.Dense(32, activation="relu")(action_out)


    #comp_input = layers.Input(shape=(num_actions))
    comp_out = layers.Dense(32, activation="relu")(action_input) 

    #action_attention = layers.Attention()([state_input[:,:, :2], action_input])
    action_convolution = layers.Conv1D(64, kernel_size=10, activation='relu')(action_input)
    #action_convolution = layers.MaxPooling1D((4))(action_convolution)
    #action_convolution = layers.Conv1D(64, kernel_size=2, activation='relu')(action_convolution)
    #action_convolution = layers.MaxPooling1D((2))(action_convolution)

    action_convolution = layers.Flatten()(action_convolution)
    #action_convolution = layers.Reshape((32))(action_convolution)
#
    #print("state", state_out, "action", action_out)
    #attention = layers.MultiAttention()([state_out, action_out])

    #attention_out = layers.Dense(64, activation="relu")(action_attention)
    
    #attention_out = layers.Dense(32)(attention_out)
    #make actions know about each other by self attention
    #action_out = layers.Reshape((None, 32))(action_out)
    #action_out = layers.Dense(32)(action_out)
    #print("action out", action_out)
    # Both are passed through seperate layer before concatenating
    #concat = layers.Concatenate()([state_out[:,:,:2], action_out, attention_out])#action_attention, action_out, close_input_attention])
    concat = layers.Concatenate()([state_out, action_convolution])
    concat = tf.keras.backend.repeat(state_out, n=10)
    concat2 = layers.Concatenate()([concat, action_out])
    #concat2 = tf.tile(concat, action_out)
    out = layers.Dense(64, activation="relu")(concat2)
    out = layers.Dense(32, activation="relu")(out)
   # out = layers.Rescaling(1/1000)(out)
    outputs = layers.Dense(1)(out)
    #print(outputs)
    # Outputs single value for give state-action
    # this is changed 
    #model = tf.keras.Model([state_input, action_input, close_input, comp_input], outputs)

    model = tf.keras.Model([state_input, action_input], outputs)
    return model
