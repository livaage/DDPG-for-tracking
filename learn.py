from noise import OUActionNoise 
from model import get_actor, get_critic
from replay import Buffer, update_target
from env import TrackEnv
from policy import policy 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
import replay 
import globals 

globals.initialise_globals() 

std_dev = 5
ou_noise1 = OUActionNoise(mean=np.array([0]), std_deviation=float(std_dev) * np.ones(1))

std_dev = 20
ou_noise2 = OUActionNoise(mean=np.array([0]), std_deviation=float(std_dev) * np.ones(1))

# Discount factor for future rewards
# Used to update target networks
tau = 0.005

buffer = Buffer(100, 64)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
total_episodes = 100000
# Takes about 4 min to train

env = TrackEnv()
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise1)

        state, reward, done = env.step(action[0])
        
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(globals.target_actor.variables, globals.actor_model.variables, tau)
        update_target(globals.target_critic.variables, globals.critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()