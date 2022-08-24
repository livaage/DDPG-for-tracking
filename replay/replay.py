import numpy as np 
import tensorflow as tf 
import yaml 
from model.actor import get_actor
from model.critic import get_critic
import pandas as pd 


with open("/home/lhv14/DDPG/config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

num_states = config['num_states']
num_actions = config['num_actions']
upper_bound = config['upper_bound']
lower_bound = config['lower_bound']


critic_lr = 0.001
actor_lr = 0.001
gamma = 0.5

#clipvalue

critic_optimizer = tf.keras.optimizers.Adam(critic_lr, amsgrad=True)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

class Buffer:
    def __init__(self, target_actor, target_critic, actor_model, critic_model,  buffer_capacity=100, batch_size=64):
        # Number of "experiences" to store at max
        
        self.target_actor = target_actor
        self.target_critic = target_critic 
        self.actor_model = actor_model 
        self.critic_model = critic_model 
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1][0]
        #self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            #tf.print(target_actions)
            #next_allowed_state_batch = find_closest_allowed_next_states(state_batch, action_batch) 
            

            y = reward_batch + gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )

            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value)) # + tf.math.square(y - tf.cast(next_state_batch[:, :2], dtype=tf.float32)))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions =self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

        #print(critic_grad, actor_grad)

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


