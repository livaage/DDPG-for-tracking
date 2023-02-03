from dataloaders.dataloader import DataLoader 
from env.trackml_newstate import TrackMLState
from env.trackml_dev import TrackMLPredposEnv
from model.actor import get_actor
#from model.critic import get_critic 
from model.single_input_critic import get_critic 
import tensorflow as tf
from utils.noise import OUActionNoise 
import numpy as np 
from replay.replay import Buffer, update_target 
from policy.policy import policy
#from utils.find_compatible_hits import Find_Compatible_Hits_ModuleMap_Line
from utils.find_compatible_hits_dev import Find_Compatible_Hits_ModuleMap_Line_New
import csv 
import pandas as pd 
import yaml

#from policy.policy_maxQ import policy_maxQ 
from policy.policy_maxQ_1 import policy_maxQ
from policy.get_comp_hits import get_comp_hits
import dowel
from dowel import logger, tabular

critic_optimizer = tf.keras.optimizers.RMSprop( learning_rate=0.001,
rho=0.9,
momentum=0.0,
epsilon=1e-07,
centered=False,
decay=0,
clipnorm=True,
clipvalue=True,
global_clipnorm=None,)


logger.add_output(dowel.TensorBoardOutput('tensorboard_logdir'))

dl = DataLoader() 
file_counter = 0 


f = open("evaluation/comp_hits_qualities.csv", "w")
writer = csv.writer(f)
writer.writerow(["cor_z", "cor_r", "comp_hit_z", "comp_hit_r", "quality", "rank", "is right", "is chosen"])

with open("/home/lhv14/GCRL/DDPG/config.yaml", "r") as k:
    config = yaml.load(k, Loader=yaml.FullLoader)

num_close_hits = config['num_close_hits']



#hits, subset_hits = dl.load_data_trackml(file_counter) 
hits, subset_hits = dl.load_data_trackml(file_counter) 
comp = Find_Compatible_Hits_ModuleMap_Line_New(hits) 

env = TrackMLState(hits, subset_hits, comp)



critic_model = get_critic() 

target_critic = get_critic() 


target_critic.set_weights(critic_model.get_weights())



# rate of updating target network
tau = 0.005
# buffer_capacity, batch_size

buffer = Buffer(target_critic, critic_model, comp, 10000, 1)
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
total_episodes = 2000000
episode_counter = 0 
reward_buffer = [] 


for ep in range(total_episodes):
    prev_state, correct_hit = env.reset()
    #print("starting state is", prev_state, "correct hit is", correct_hit)
    episodic_reward = 0
    reward_buffer = [] 
    #episode_counter = 0 
    done = False
    while True:

        #print("!!!  new round   !!!")
        #print("correct hit given to comp hits", correct_hit)
        comp_hits, rewards, cor = get_comp_hits(prev_state, comp, correct_hit)

        #print("wegiht", critic_model.get_weights()[-2])

        action, return_best_action, q_vals = policy_maxQ(prev_state, critic_model, comp_hits)
        #print("q values are", q_vals, "return best action is ", return_best_action, "with action", action, "correct hit is ", comp.get_hit(cor)[['z', 'r']])
        #print("comp hits are", comp_hits)
        prev_correct_hit = correct_hit
        state, reward, correct_hit, done = env.step(action, correct_hit, done)
        #print("correct hit from step", comp.get_hit(correct_hit)[['z', 'r']], "reward is ", reward)

        #print("action has said that next state is", state, "with correct hit ", correct_hit)
        #print("to buffer writing in action ", action, "and correct hit", comp.get_hit(correct_hit)[['z', 'r', 'hit_id']])
        #print("to buffer writing:", "prev_state", prev_state, "action", action, "rewards", rewards, "state", state, "q_vals", q_vals, "comp_hits", comp_hits, "prev cor hit", prev_correct_hit)
        sorted_qvals = np.sort(q_vals)

        cor_row = comp.get_hit(cor).squeeze()

        for i in range(len(comp_hits)):
            #print(np.where(sorted_qvals == q_vals[i])[0])
            is_chosen = (q_vals[i] == return_best_action)
            is_right = (comp_hits[i][1] == cor_row.r)
            comp_row = comp_hits[i]
            # if ((is_right) & (is_chosen)): 
            #     print(best_action,  cor[['z', 'r']])
            #print(is_right, best_action, cor[['z', 'r']])
            row = pd.DataFrame({
            'cor_z': [cor_row['z']],
            'cor_r': [cor_row['r']],  
            'comp_hit_z': [comp_row[0]], 
            'comp_hit_r': [comp_row[1]], 
            'reward': [rewards[i]],
            'quality': q_vals[i], 
            'rank': [np.where(sorted_qvals == q_vals[i])[0]], 
            'is right': [is_right], 
            'is chosen': [is_chosen]})
            row.to_csv(f, mode='a', header=None, index=None)
        
        
        #print(prev_state, action, reward, state, q_vals, comp_hits, prev_correct_hit, correct_hit)
        buffer.record((prev_state, action, reward, state, q_vals, comp_hits, prev_correct_hit, correct_hit))
        episodic_reward += reward
        reward_buffer.append(reward)

        #critic_loss = buffer.learn()
        #print(critic_loss)

        comp_all_batch= np.tile(comp_hits, (num_close_hits,1)).reshape((-1, num_close_hits, 2))
        input = [tf.convert_to_tensor(np.tile(state, (num_close_hits, 1))), tf.convert_to_tensor(comp_hits), tf.convert_to_tensor(comp_all_batch)]
   
        with tf.GradientTape() as tape:

            #print("y is ", y, "reward is ", reward_batch)
            critic_value = critic_model(input, training=True)
            #print("input", input)
            critic_value = tf.squeeze(critic_value)

            critic_loss = tf.math.square(rewards - critic_value)
            #print("critic loss", critic_loss)
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        #print(critic_grad)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )


        tabular.record('loss', critic_loss.numpy())
        #update_target(target_critic.variables, critic_model.variables, tau)
        tabular.record('hit reward', reward)




        # End this episode when `done` is True
        if done:
            break
        prev_state = state

        if episode_counter > 400: 
            episode_counter = 0 
            file_counter += 1 
            hits, allowed_pids = dl.load_data_trackml(file_counter) 
            comp = Find_Compatible_Hits_ModuleMap_Line_New(hits) 
            env = TrackMLState(hits, allowed_pids, comp)
            #buffer = Buffer(target_critic, critic_model, comp, 10000, 8)
            print("new file!")
            break

    ep_reward_list.append(episodic_reward)
    #print("reward buffer", reward_buffer)
    tabular.record('episodic_reward', episodic_reward)
    #rew = np.array(reward_buffer[-40:])

    accuracy =  len(np.where(np.round(np.array(reward_buffer), 4) == 0)[0]) / len(reward_buffer)
    
    tabular.record('accuracy', accuracy)
    logger.log(tabular)
    logger.dump_all()
    episode_counter += 1 





    # Mean of last 40 episodes
    # avg_reward = np.mean(ep_reward_list[-40:])
    # rew = np.array(reward_buffer[-40:])
    # accuracy = len(np.where(rew == 0)[0]) / len(rew)
    # print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    # avg_reward_list.append(avg_reward)
    # print("average hit accuracy is", accuracy)

    print("Episode * {} * Reward is ==> {}".format(ep, episodic_reward))
    #print("accuracy ", accuracy)

