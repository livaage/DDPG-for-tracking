"""Similar to the regular trackmlenv, but this time predicts position and not a correction!."""
from concurrent.futures import process
import math
from os import preadv

import akro
#from charset_normalizer import detect
import numpy as np
import circle_fit as cf
import pandas as pd 
#from garage import Environment, EnvSpec, EnvStep, StepType

import random 
from gym.spaces import Box
import csv
import trackml.dataset
import json
import yaml

from utils.geometry import find_m_b
from utils.hits import find_df_hits

f = open("evaluation/garage_outputs.csv", "w")
writer = csv.writer(f)
writer.writerow(["filenumber", "particle_id", "mc_z", "mc_r", "pred_z", "pred_r", "reward"])

class TrackMLState():
    """A simple 2D point environment.

    Args:
        goal (np.ndarray): A 2D array representing the goal position
        arena_size (float): The size of arena where the point is constrained
            within (-arena_size, arena_size) in each dimension
        done_bonus (float): A numerical bonus added to the reward
            once the point as reached the goal
        never_done (bool): Never send a `done` signal, even if the
            agent achieves the goal
        max_episode_length (int): The maximum steps allowed for an episode.

    """

    def __init__(self, hits, subset_hits):
        self._total_step_cnt = 0 
        self._step_cnt = None
        self.event = hits 
        # hit and previous hit is observation 
        self._observation_space = akro.Box(low=np.array([-300, 0, -1000, -1000, -100, -100]), high=np.array([300, 120, 1000, 1000, 100, 100]), dtype=np.float64)
        self._action_space = akro.Box(low=np.array([-300, 0]),
                                      high=np.array([300,120]),
                                      shape=(2, ),
                                      dtype=np.float32)

        self.record_z = [] 
        self.record_r = []
        self.record_pid = []
        self.record_event_counter = [] 
        self.record_reward = [] 
        self.record_a0 = [] 
        self.record_a1 = [] 
        self.record_filenumber = [] 
        self.prev_three_layer_buffer = [0,0,0] 
        self.dm = 0 
        self.subset_hits = subset_hits
        self.allowed_pids = subset_hits['particle_id'].unique() 


    def reset(self):
        """Reset the environment.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditisoned or MTRL.)

        """
      

        random_particle_id = random.choice(self.allowed_pids)
        #first_particle_id = self.event.particle_id.iloc[0]
        
        #self.particle = self.event[self.event['particle_id']==first_particle_id]
        
        self.particle = self.subset_hits[self.subset_hits['particle_id']==random_particle_id]
        #print(self.event, random_particle_id)
        #print(self.particle)
        self.original_pid = random_particle_id
        #self.original_pid = first_particle_id

        
        start_hit = self.particle.iloc[0,:]
        self.start_hit = start_hit 
        next_hit = self.particle.iloc[1,:]#
        self.num_track_hits = 1


        if start_hit.unique_layer_id == next_hit.unique_layer_id: 
            next_hit = self.particle.iloc[2, :]
            self.num_track_hits = 2

        m, b = find_m_b(start_hit, next_hit)
        self.dm = m 
        self.prev_three_layer_buffer[0] = start_hit.unique_layer_id
        self.prev_three_layer_buffer[1] = next_hit.unique_layer_id
        self.prev_three_layer_buffer[2] = next_hit.unique_layer_id

   
        self._point = next_hit[['z', 'r']].values 

      
        self.max_r = next_hit.r
        #fix
        self.file_counter = 0 

        #prepare file to write output 
        # row = pd.DataFrame({'filenumber': [self.file_counter, self.file_counter], 
        # 'particle_id': [self.original_pid, self.original_pid], 
        # 'mc_z': [start_hit.z, next_hit.z],
        # 'mc_r' : [start_hit.r, next_hit.r],
        # 'reward': [0, 0 ]})
        # row.to_csv(f, mode='a', header=None, index=None)
        
        self.pt = self.start_hit.pt


        self.state = [self._point[0], self._point[1], m, b, self.pt]
        observation = self.state 


        self.previous_correct_hit = next_hit 

        self._step_cnt = 0
        self.original_particle = self.subset_hits[self.subset_hits['particle_id']==self.original_pid].reset_index()

        return observation

    def step(self, action, done):
        """Step the environment.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            EnvStep: The environment step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment
            has been
                constructed and `reset()` has not been called.

        """

        signal_done = done 
        if self._step_cnt is None:
            raise RuntimeError('reset() must be called before step()!')
        
        #action = action*1000
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        #print("action is", action)
        #redundant when already have only allowed points 
        # a_clipped = np.clip(a, self._action_space.low, self._action_space.high)

        # #already clipped, change 
        # predicted_point_z = np.clip(a_clipped[0],  -300, 300)
        # predicted_point_r = np.clip(a_clipped[1], 0, 120)


        #predicted_point = [a_clipped[0], a_clipped[1]]
        #print("predicted point", predicted_point)
        predicted_point = a
        #print(a)
        self.previous_state = self.state


        hit1_df = self.event[(self.event['z'] == self.previous_state[0]) & (self.event['r'] == self.previous_state[1])]
        hit2_df = self.event[(self.event['z'] == predicted_point[0]) & (self.event['r'] == predicted_point[1])]
        
        #why is this needed 
        if hit1_df.shape[0] != 1: 
            hit1_df = hit1_df.iloc[0]
        if hit2_df.shape[0] != 1: 
            hit2_df = hit2_df.iloc[0]
        
        # can be removed? 
        hit1_df = hit1_df.squeeze()
        hit2_df = hit2_df.squeeze() 
        self.prev_three_layer_buffer[0] = self.prev_three_layer_buffer[1]
        self.prev_three_layer_buffer[1] = self.prev_three_layer_buffer[2]
        try: 
            self.prev_three_layer_buffer[2] = hit2_df.unique_layer_id
        except: 
            print("this werird issue hit 2 is ", predicted_point, "hit2_df ", self.event[(self.event['z'] == predicted_point[0]) & (self.event['r'] == predicted_point[1])], 
            "shape is ", self.event[(self.event['z'] == predicted_point[0]) & (self.event['r'] == predicted_point[1])].shape)
            print("this werird issue hit 1 is ", "hit1_df ", self.event[(self.event['z'] == self.previous_state[0]) & (self.event['r'] == self.previous_state[1])], 
            "shape is ", self.event[(self.event['z'] == self.previous_state[0]) & (self.event['r'] == self.previous_state[1])].shape)
        
        
        if hit1_df.unique_layer_id != hit2_df.unique_layer_id: 
            m, b = find_m_b(hit1_df, hit2_df)
        else: 
            m, b = self.previous_state[2], self.previous_state[3] 
        

        # this cna me zero 
        self.dm = m - self.previous_state[2] 
        self.db = b - self.previous_state[3]


        self.state = [a[0], a[1], m, b, self.pt] 
        
        op_wo_self = self.original_particle[self.original_particle['unique_layer_id']!=self.previous_correct_hit.unique_layer_id]
        try: 
            correct_hit = op_wo_self[(op_wo_self['r'] > self.previous_state[1]) & (op_wo_self['r'] > self.max_r)].iloc[0]
            end_track = False
        except: 
            correct_hit = self.previous_correct_hit 
            end_track = True 
        
        self.max_r = correct_hit.r 
        distance = np.sqrt((predicted_point[0]-correct_hit.z)**2 + (predicted_point[1]-correct_hit.r)**2)
        
        if distance == 0: 
            reward = 10
        elif end_track: 
            reward = 0
        elif self.previous_state[1] == self.state[1]: 
            reward = -5
        else: 
            reward = -distance 
        
        self.previous_correct_hit = correct_hit
        
        next_hit = correct_hit 
        # try:
        #     next_hit = self.original_particle.iloc[self.num_track_hits+1]
        # except: 
        #     next_hit = self.original_particle.iloc[-1]
        
        
        # distance = np.sqrt((predicted_point[0]-next_hit.z)**2 + (predicted_point[1]-next_hit.r)**2)
        # reward = - distance
        # if (hit1_df.unique_layer_id == hit2_df.unique_layer_id) & (len(self.original_particle[self.original_particle['unique_layer_id']==hit2_df.unique_layer_id])) < 2: 
        #    self.num_track_hits += 1 
        

        # hit_layer = np.unique(self.event[(self.event['z'] == predicted_point_z) & (self.event['r'] == predicted_point_r)].unique_layer_id)[0]
        #         #print(hit_layer)
        # correct_hit_inthatlayer = self.original_particle[(self.original_particle['unique_layer_id']==hit_layer) #& (
        #             #self.original_particle['r'] > self.previous_correct_hit.r)
        #         ]
        # #no_hit_in_layer = False





        #         #next_index = self.num_track_hits + 1 
        #         #if next_index > len(self.original_particle) -1: 
        #         #     next_index = len(self.original_particle) - 1
        #         #prev_index = self.original_particle[self.original_particle['hit_id']==hit1_df.hit_id].index[0]
        # po = self.original_particle[self.original_particle['r'] > self.previous_correct_hit.r]
        # try: 
        #     next_layer_hit = po.iloc[0]
        # except: 
        #     #print("excepting here, po is", po, "and the prev thing is ", self.previous_correct_hit.r, "original aprticle", self.original_particle )
        #     next_layer_hit = self.original_particle.iloc[-1]
        #     done = True
        
        
        # if (len(correct_hit_inthatlayer) > 1): 
        #     distance = np.sqrt((predicted_point[0]-correct_hit_inthatlayer.z.values)**2 + (predicted_point[1]-correct_hit_inthatlayer.r.values)**2)
        #     next_hit = correct_hit_inthatlayer.iloc[np.argmin(distance)].squeeze()
        #     #print("in first ", next_hit.z)
        #     distance = np.min(distance)

        #     #print(distance)
        # # elif  (self.prev_three_layer_buffer[0] == self.prev_three_layer_buffer[1] == self.prev_three_layer_buffer[2]): 
        # #    # distance = np.sqrt((predicted_point[0]-correct_hit_inthatlayer.z.values)**2 + (predicted_point[1]-correct_hit_inthatlayer.r.values)**2)
        # #    # next_hit = correct_hit_inthatlayer.iloc[np.argmin(distance)].squeeze()
        # #     next_hit = self.original_particle.loc[next_index,: ]
        # #     distance = 10

        # elif len(correct_hit_inthatlayer) ==1: 
        #     next_hit = correct_hit_inthatlayer.squeeze() 
        #     distance = np.sqrt((predicted_point[0]-next_hit.z)**2 + (predicted_point[1]-next_hit.r)**2)
        #     #print(distance)
        # elif predicted_point_r < np.max(self.original_particle.r): 
        #     next_hit = next_layer_hit
        #     distance = np.sqrt((predicted_point[0]-next_hit.z)**2 + (predicted_point[1]-next_hit.r)**2)

        #     # no hits in that layer
        #     # next_index = self.num_track_hits + 1 

        #     # if next_index > len(self.original_particle) -1: 
        #     #     next_index = len(self.original_particle) - 1
        #     #     done = True 
        #     # next_hit = self.original_particle.loc[next_index,: ]
            
        #     # if next_hit.r < self.previous_state[1]:
        #     #     next_hit = next_layer_hit
        #     # no_hit_in_layer = True

        #     # if predicted_point_r < np.max(self.original_particle.r): 
        #     #     distance  = 3 
        # else: 
        #     next_hit = self.original_particle.iloc[-1]
        #     distance = 100 
        #     done = True
        # done_ep = False
        # if next_hit.r == np.max(self.original_particle.r): 
        #     done_ep = True
        # self.previous_correct_hit = next_hit
        
        # distance = np.min(np.sqrt((predicted_point[0]-self.original_particle.z)**2 + (predicted_point[1]-self.original_particle.r)**2))
        # distance = np.sqrt((predicted_point[0]-next_hit.z)**2 + (predicted_point[1]-next_hit.r)**2)
    
        # if distance ==0: 
        #     reward = 10
        # elif distance ==100: 
        #     reward = 0 
        # else: 
        #     reward = -distance
        #     #reward = -10

        # gone_beyond = False
 

        # changing to reward is just distance to closest hit that is in track 
        

        #if (np.abs(self.previous_state[1] - predicted_point[1]) <1)  & (np.abs(reward) > 10): 
        #    self.num_track_hits -= 1
  
        self.num_track_hits += 1 
    
       
      
        self._step_cnt += 1
        self._total_step_cnt += 1
  

        row = pd.DataFrame({'filenumber': [self.file_counter], 
        'particle_id': [self.original_pid], 
        'mc_z': [next_hit.z], 
        'mc_r' : [next_hit.r], 
        'pred_z': [a[0]], 
        'pred_r': [a[1]], 
        'reward': [reward] })
        row.to_csv(f, mode='a', header=None, index=None)
        num_layer_hits = self.original_particle.unique_layer_id.nunique()

        #if predicted point is above, there are no compatible hits and already predicted 7 hits or 
        #if (predicted_point_r > np.max(self.original_particle.r)) or (signal_done & (self.num_track_hits > 7)) :#or (done_ep) or ):
        #or (done==True) or (
        #    no_hit_in_layer &  self.num_track_hits > num_layer_hits): 
        if  self.num_track_hits > 7 or end_track:

            done = True 
        else: 
            done = False 
            #self.episode_counter +=1 

        self._point = predicted_point
        #self.state = [self._point[0], self._point[1] , m, b] # self.dm, self.db]
        observation = self.state


       # self.average_reward = (self.average_reward + reward)/2


        return observation, reward, done


    def close(self):
        """Close the env."""

    # pylint: disable=no-self-use
    def dump_summary(self):
        print("dr:   ", "\n dz:    " ) 



#  hit_layer = np.unique(self.event[(self.event['z'] == predicted_point_z) & (self.event['r'] == predicted_point_r)].unique_layer_id)[0]
#         #print(hit_layer)
#         correct_hit_inthatlayer = self.original_particle[(self.original_particle['unique_layer_id']==hit_layer) #& (
#             #self.original_particle['r'] > self.previous_correct_hit.r)
#         ]
#         no_hit_in_layer = False





#         #next_index = self.num_track_hits + 1 
#         #if next_index > len(self.original_particle) -1: 
#         #     next_index = len(self.original_particle) - 1
#         #prev_index = self.original_particle[self.original_particle['hit_id']==hit1_df.hit_id].index[0]
#         po = self.original_particle[self.original_particle['r'] > self.previous_correct_hit.r]
#         try: 
#             next_layer_hit = po.iloc[0]
#         except: 
#             #print("excepting here, po is", po, "and the prev thing is ", self.previous_correct_hit.r, "original aprticle", self.original_particle )
#             next_layer_hit = self.original_particle.iloc[-1]
#             done = True
        
        
#         if (len(correct_hit_inthatlayer) > 1): 
#             distance = np.sqrt((predicted_point[0]-correct_hit_inthatlayer.z.values)**2 + (predicted_point[1]-correct_hit_inthatlayer.r.values)**2)
#             next_hit = correct_hit_inthatlayer.iloc[np.argmin(distance)].squeeze()
#             #print("in first ", next_hit.z)
#             distance = np.min(distance)

#             #print(distance)
#         # elif  (self.prev_three_layer_buffer[0] == self.prev_three_layer_buffer[1] == self.prev_three_layer_buffer[2]): 
#         #    # distance = np.sqrt((predicted_point[0]-correct_hit_inthatlayer.z.values)**2 + (predicted_point[1]-correct_hit_inthatlayer.r.values)**2)
#         #    # next_hit = correct_hit_inthatlayer.iloc[np.argmin(distance)].squeeze()
#         #     next_hit = self.original_particle.loc[next_index,: ]
#         #     distance = 10

#         elif len(correct_hit_inthatlayer) ==1: 
#             next_hit = correct_hit_inthatlayer.squeeze() 
#             distance = np.sqrt((predicted_point[0]-next_hit.z)**2 + (predicted_point[1]-next_hit.r)**2)
#             #print(distance)
#         elif predicted_point_r < np.max(self.original_particle.r): 
#             next_hit = next_layer_hit
#             distance = np.sqrt((predicted_point[0]-next_hit.z)**2 + (predicted_point[1]-next_hit.r)**2)

#             # no hits in that layer
#             # next_index = self.num_track_hits + 1 

#             # if next_index > len(self.original_particle) -1: 
#             #     next_index = len(self.original_particle) - 1
#             #     done = True 
#             # next_hit = self.original_particle.loc[next_index,: ]
            
#             # if next_hit.r < self.previous_state[1]:
#             #     next_hit = next_layer_hit
#             # no_hit_in_layer = True

#             # if predicted_point_r < np.max(self.original_particle.r): 
#             #     distance  = 3 
#         else: 
#             next_hit = self.original_particle.iloc[-1]
#             distance = 100 
#             done = True