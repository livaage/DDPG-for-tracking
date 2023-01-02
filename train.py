from dataloaders.dataloader import DataLoader 
from env.trackml_newstate import TrackMLState
from env.trackml_dev import TrackMLPredposEnv
from model.actor import get_actor
from model.critic import get_critic 
import tensorflow as tf
from utils.noise import OUActionNoise 
import numpy as np 
from replay.replay import Buffer, update_target 
from policy.policy import policy
#from utils.find_compatible_hits import Find_Compatible_Hits_ModuleMap_Line
from utils.find_compatible_hits_dev import Find_Compatible_Hits_ModuleMap_Line_New

from policy.policy_maxQ import policy_maxQ 
import dowel
from dowel import logger, tabular


logger.add_output(dowel.TensorBoardOutput('tensorboard_logdir'))

dl = DataLoader() 
file_counter = 0 


hits, subset_hits = dl.load_data_trackml(file_counter) 
env = TrackMLState(hits, subset_hits)
comp = Find_Compatible_Hits_ModuleMap_Line_New(hits) 



actor_model = get_actor() 
critic_model = get_critic() 

target_actor = get_actor() 
target_critic = get_critic() 


target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())


# Learning rate for actor-critic models

 
ou_noise1 = OUActionNoise(mean=np.array([0]), std_deviation=float(10) * np.ones(1))
ou_noise2 = OUActionNoise(mean=np.array([0]), std_deviation=float(6) * np.ones(1))

# rate of updating target network
tau = 0.05

# buffer_capacity, batch_size

buffer = Buffer(target_actor, target_critic, actor_model, critic_model, comp, 128, 8)
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
total_episodes = 2000000
episode_counter = 0 
reward_buffer = [] 


for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0
    reward_buffer = [] 
    #episode_counter = 0 

    while True:
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        #action = policy(tf_prev_state, ou_noise1, ou_noise2, actor_model, comp)
        action, return_action, state_desc, q_vals, done = policy_maxQ(prev_state, target_critic, comp)
        #print(action, done)
        state, reward, done = env.step(action, done)
        # action[0] if other policy
        
        buffer.record((prev_state, action, reward, state, q_vals, state_desc[1]))
        episodic_reward += reward
        reward_buffer.append(reward)

        buffer.learn()
        #update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)
        tabular.record('hit reward', reward)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state
        if episode_counter > 1000: 
            episode_counter = 0 
            file_counter += 1 
            hits, allowed_pids = dl.load_data_trackml(file_counter) 
            comp = Find_Compatible_Hits_ModuleMap_Line_New(hits) 
            env = TrackMLState(hits, allowed_pids)
            buffer = Buffer(target_actor, target_critic, actor_model, critic_model, comp, 100, 8)
            print("new file!")
            break

    ep_reward_list.append(episodic_reward)
    tabular.record('episodic_reward', episodic_reward)
    rew = np.array(reward_buffer[-40:])

    accuracy =  len(np.where(rew == 10)[0]) / len(rew)
    tabular.record('accuracy', accuracy)
    logger.log(tabular)
    logger.dump_all()
    episode_counter += 1 





    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    rew = np.array(reward_buffer[-40:])
    accuracy = len(np.where(rew == 10)[0]) / len(rew)
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)
    print("average hit accuracy is", accuracy)

