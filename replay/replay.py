import numpy as np 
import tensorflow as tf 
import yaml 
#from model.actor import get_actor
#from model.critic import get_critic
import pandas as pd 
from policy.policy_maxQ_1 import policy_maxQ
from policy.get_comp_hits import get_comp_hits
from utils.geometry import find_m_b_no_df
import csv 

tf.config.run_functions_eagerly(True)

with open("/home/lhv14/GCRL/DDPG/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

num_states = config['num_states']
num_actions = config['num_actions']
upper_bound = config['upper_bound']
lower_bound = config['lower_bound']
num_close_hits = config['num_close_hits']
critic_lr = 0.001
#actor_lr = 0.001
gamma = 0.99

#clipvalue


critic_optimizer = tf.keras.optimizers.Adam(critic_lr, amsgrad=False)

# critic_optimizer = tf.keras.optimizers.RMSprop( learning_rate=1e-3,
#     rho=0.9,
#     momentum=0.0,
#     epsilon=1e-07,
#     centered=False,
#     decay=0,
#     clipnorm=True,
#     clipvalue=True,
#     global_clipnorm=None,)

f = open("evaluation/replay_check.csv", "w")
writer = csv.writer(f)
writer.writerow(["hit1_r", "hit2_r", "hit3_r", "qvals", "maxq"])



#actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

class Buffer:
    def __init__(self, target_critic, critic_model,  comp, buffer_capacity=100, batch_size=64):
        # Number of "experiences" to store at max
        
        #self.target_actor = target_actor
        self.target_critic = target_critic 
        #self.actor_model = actor_model 
        self.critic_model = critic_model 
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.comp_hits_buffer = np.zeros((self.buffer_capacity, num_close_hits,num_actions)) 
        self.nearby_hits_buffer = np.zeros((self.buffer_capacity, num_close_hits))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, num_close_hits))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.qvals_buffer = np.zeros((self.buffer_capacity, num_close_hits))
        self.correct_hit_buffer = np.zeros(self.buffer_capacity)
        self.prev_state_buffer = np.zeros((self.buffer_capacity, num_states))
        #self.close_buffer = np.zeros((self.buffer_capacity, 11, num_actions))
        #self.state_desc_buffer = np.zeros((self.buffer_capacity, 3, 10))
        self.comp = comp

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] =  obs_tuple[0]
        #self.action_buffer[index] = obs_tuple[1][0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        #self.next_state_buffer[index] = obs_tuple[3]
        try: 
            self.next_state_buffer[index] = obs_tuple[3]
        except: 
            print("obs tuple is ", obs_tuple, "index is ", index)
        #self.close_buffer[index] = obs_tuple[4]
        self.buffer_counter += 1
        self.qvals_buffer[index] = obs_tuple[4]
        self.comp_hits_buffer[index] = obs_tuple[5]
        self.correct_hit_buffer[index] = obs_tuple[6]
        self.prev_state_buffer[index] = obs_tuple[7]
    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, comp_hits_batch, correct_hit_batch, prev_hit_batch
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:

            # q_vals_batches = []
            # next_comp_hits = np.zeros((len(state_batch), num_close_hits, 2)) 
            # for i in range(len(next_state_batch)): 
            #     comp_hits_i, rewards, cor = get_comp_hits(next_state_batch[i], self.comp, correct_hit_batch[i])
            #     next_comp_hits[i] =  comp_hits_i

            next_comp_hits =  np.zeros((comp_hits_batch.shape[0], comp_hits_batch.shape[1], num_close_hits, 2))
            
            #for each starting hit 
            # max = []
            # for i in range(len(comp_hits_batch)): 
            #     # for each of the compatible hits 
            #     qs = []
            #     for j in range(len(comp_hits_batch[i])):
            #         #print("i'm in the buffer")
            #         m, b = find_m_b_no_df(state_batch[i], comp_hits_batch[i][j])
            #         comp_hits_i, rewards, cor = get_comp_hits(comp_hits_batch[i][j], self.comp, correct_hit_batch[i])
            #         potential_state = [comp_hits_batch[i][j][0], comp_hits_batch[i][j][1], m, b, comp_hits_batch[i][j][2], comp_hits_batch[i][j][3]]
            #         potential_state = np.tile(potential_state, (num_close_hits, 1)) 
            #         comp_all = np.tile(comp_hits_i, (num_close_hits,1)).reshape((-1, num_close_hits, 4))
            #         #comp = tf.reshape(comp_hits_batch[i][j], (-1, num_close_hits, 4))
            #         model_input = [tf.convert_to_tensor(potential_state), tf.convert_to_tensor(comp_hits_i), tf.convert_to_tensor(comp_all)]
            #         #print(model_input)
            #         q_vals = self.target_critic(model_input, training=True)
            #         qs.append(np.max(q_vals))
            #     max.append(qs)  


    #for i in comp_hits_z_r: 
       # model_input = [tf.convert_to_tensor(state), tf.convert_to_tensor(i)]

    #model_input = [tf.expand_dims(state, axis=1), tf.expand_dims(i, axis=1)]
        #print("policy input", model_input)
    #print("model input", model_input)
        #print("calling in loop")
        #print([state, i])
        # tf.reshape(input_tensor,shape=(1,n))    # n is the number of samples, feature tensor have 
        #print("reshaped", np.array(state).reshape(1,4))
        #q_vals.append(critic([np.array(state[:2]).reshape(1,2), np.array(i).reshape(1, 2), comp_hits_z_r.reshape(1, num_close_hits, 2)])) 
    #print(q_vals)from keras.utils import to_categorical 
    #q_vals = critic(model_input)    

                    #print("potential state", potential_state)
                    # is that the right correct hit 
                    # find the next correct hits for that starting point and its compatible hit - doens't matter 
                    #new_comp_hits, rewards, cor = get_comp_hits(potential_state, self.comp, correct_hit_batch[i], dope=True)
                    #next_comp_hits[i][j] =  new_comp_hits
                    #print(new_comp_hits.shape)
                    #action, return_best_action, q_vals = policy_maxQ(potential_state, self.target_critic, new_comp_hits)

     #               max.append(tf.reduce_max(q_vals)) 

                    #print("hit1 is", state_batch[i], "hit 2 is", comp_hits_batch[i][j], "hit 3s are", new_comp_hits, "qvals are", q_vals, "max", tf.reduce_max(q_vals))
                    # for k in range(len(new_comp_hits)): 
                    #     row = pd.DataFrame({
                    #     'hit1_r': [state_batch[i][1].numpy()],
                    #     'hit2_r': [comp_hits_batch[i][j][1].numpy()],  
                    #     'hit3_r': [new_comp_hits[k]], 
                    #     'qvals': [q_vals[k]], 
                    #     #'reward': [rewards[i]],
                    #     #'quality': q_vals[i], 
                    #     'maxq': [tf.reduce_max(q_vals).numpy()]})
                    #     row.to_csv(f, mode='a', header=None, index=None)



            #max = tf.reshape(tf.stack(max), (self.batch_size, num_close_hits))
            #print("!!!next comp hits" , next_comp_hits)
            
           # next_comp_hits = next_comp_hits.reshape(self.batch_size*num_close_hits*num_close_hits, 2)
            comp_hits_batch = tf.reshape(comp_hits_batch, (self.batch_size*num_close_hits, num_actions))
            #print(next_state_batch, next_comp_hits)
            #action, return_best_action, q_vals = policy_maxQ(next_state_batch, self.target_critic, next_comp_hits)
            #print(q_vals)
            #print(reward_batch) #tf.math.reduce_max(reward_batch, axis=1))

            #print("reward batch", reward_batch)
            #y = tf.reshape(reward_batch, self.batch_size*num_close_hits) + gamma * q_vals


            y = reward_batch #+ gamma*max
            y = tf.reshape(y, self.batch_size*num_close_hits)

            #print(y)
            comp_all_batch= np.tile(comp_hits_batch, (num_close_hits,1)).reshape((-1, num_close_hits, num_actions))
            input = [tf.convert_to_tensor(np.tile(state_batch, (num_close_hits, 1))), tf.convert_to_tensor(comp_hits_batch), tf.convert_to_tensor(comp_all_batch)]
            critic_value = self.critic_model(input, training=True)
            #print("input", input)

            #print(critic_value.shape)
            
            #critic_value = tf.linalg.normalize(critic_value)[0]
            #critic_value = tf.math.log(critic_value)
            #print("normalized ", critic_value.shape)
            #where_highest = tf.math.argmax(critic_value, axis=1)
            #print("high", where_highest)
            #critic_value = np.zeros(5)
            #critic_value[where_highest] = 1
            #critic_value = tf.convert_to_tensor(critic_value)
            #print("critic value", critic_value)
            #print("y", y)
            critic_value = tf.squeeze(critic_value)

            #print("y is ", y, "reward is ", reward_batch)
            #tf.math.reduce_mean(
            #critic_loss = tf.reduce_mean(tf.math.square(tf.reshape(y, self.batch_size*num_close_hits) - critic_value))
            #critic_loss = tf.keras.metrics.binary_crossentropy()
            h = tf.keras.losses.BinaryFocalCrossentropy()
            critic_loss = h(y, critic_value)
            #print(critic_loss)
            #print(critic_loss)
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )
        return critic_loss 

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        #print(batch_indices, self.state_buffer, self.action_buffer, self.comp_hits_buffer)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)

        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        #print("comp hits here is", self.comp_hits_buffer)
        comp_hits_batch = tf.convert_to_tensor(self.comp_hits_buffer[batch_indices])
        correct_hit_batch = tf.convert_to_tensor(self.correct_hit_buffer[batch_indices])
        prev_hit_batch = tf.convert_to_tensor(self.prev_state_buffer[batch_indices])

        critic_loss = self.update(state_batch, action_batch, reward_batch, next_state_batch, comp_hits_batch, correct_hit_batch, prev_hit_batch)
        return critic_loss

# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


