from dataloader.dataloader import DataLoader 
from env.trackml_dev import TrackMLPredposEnv 
from model.actor import get_actor
from model.critic import get_critic 
import tensorflow as tf
from utils.noise import OUActionNoise 
import numpy as np 
from replay.replay import Buffer, update_target 
from policy.policy import policy
from utils.find_compatible_hits import Find_Compatible_Hits_ModuleMap_Line


dl = DataLoader() 

hits = dl.load_data(1) 
env = TrackMLPredposEnv(hits)
comp = Find_Compatible_Hits_ModuleMap_Line(hits) 



actor_model = get_actor() 
critic_model = get_critic() 

target_actor = get_actor() 
target_critic = get_critic() 


target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())


# Learning rate for actor-critic models

 
ou_noise1 = OUActionNoise(mean=np.array([0]), std_deviation=float(10) * np.ones(1))
ou_noise2 = OUActionNoise(mean=np.array([0]), std_deviation=float(6) * np.ones(1))

tau = 0.05

buffer = Buffer(target_actor, target_critic, actor_model, critic_model, 10000, 256)
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
total_episodes = 20000

for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise1, ou_noise2, actor_model, comp)
        state, reward, done = env.step(action[0])
        
        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)
