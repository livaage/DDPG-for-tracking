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

    def __init__(self, hits, subset_hits, comp):
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
        self.comp = comp

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
        #print("the particle is ", self.particle[['z', 'r', 'unique_layer_id', 'hit_id']])
        #print(self.event, random_particle_id)
        #print(self.particle)
        self.original_pid = random_particle_id
        #self.original_pid = first_particle_id

        
        start_hit = self.particle.iloc[0,:]
        self.start_hit = start_hit 
        next_hit = self.particle.iloc[1,:]#
        self.num_track_hits = 2


        # if start_hit.unique_layer_id == next_hit.unique_layer_id: 
        #     next_hit = self.particle.iloc[2, :]
        #     self.num_track_hits = 2

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
        row = pd.DataFrame({'filenumber': [self.file_counter, self.file_counter], 
        'particle_id': [self.original_pid, self.original_pid], 
        'mc_z': [start_hit.z, next_hit.z],
        'mc_r' : [start_hit.r, next_hit.r],
        'reward': [0, 0 ]})
        row.to_csv(f, mode='a', header=None, index=None)
        
        self.pt = self.start_hit.pt


        self.state = [self._point[0], self._point[1], m, b]
        observation = self.state 

        self.previous_state = next_hit[['z', 'r']]
        self.previous_correct_hit = next_hit 

        self._step_cnt = 0
        self.original_particle = self.subset_hits[self.subset_hits['particle_id']==self.original_pid].reset_index()

        #self.previous_correct_hit = self.cor 
        self.cor = self.correct_hit()


        return observation, self.cor.hit_id

    def step(self, action, cor, done):
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



        # hit1_df = self.event[(self.event['z'] == self.previous_state[0]) & (self.event['r'] == self.previous_state[1])]
        # hit2_df = self.event[(self.event['z'] == a[0]) & (self.event['r'] == a[1])]


        # if hit1_df.shape[0] != 1: 
        #     #print(hit1_df)
        #     hit1_df = hit1_df.iloc[0]
        # if hit2_df.shape[0] != 1: 
        #     try: 
        #         hit2_df = hit2_df.iloc[0]
        #     except: 
        #         print(hit2_df, a, len(self.event))
        
        # hit1_df = hit1_df.squeeze()
        # hit2_df = hit2_df.squeeze() 
   
        hit1_df = self.comp.hit_df(self.previous_state)
        hit2_df = self.comp.hit_df(a)
        #print(hit1_df['z'].values == hit2_df['z'].values)
        m, b = find_m_b(hit1_df, hit2_df)

        self.state = [a[0], a[1], m, b] 

        correct_hit = self.cor

        #print("inside step the hit is ", a, "and the correct hit is ", correct_hit)        
        reward = self.comp.get_reward(hit2_df, correct_hit)


        #print("in step the suggested hit is", a, "and correct hit is ", self.cor)

        # #self.max_r = correct_hit.r 
        # distance = np.sqrt((a[0]-correct_hit.z)**2 + (a[1]-correct_hit.r)**2)

        # if hit2_df.hit_id == correct_hit.hit_id: 
        #     reward = 10
        # elif end_track: 
        #     reward = 0
        # # elif self.previous_state[1] == self.state[1]: 
        # #     reward = -5
        # else: 
        #     reward = -distance
        
        self.previous_correct_hit = correct_hit
        self.previous_state = self.state
        
        next_hit = correct_hit 
       
  
        self.num_track_hits += 1 
        self.cor = self.correct_hit() 
       
      
        self._step_cnt += 1
        self._total_step_cnt += 1
  

        row = pd.DataFrame({'filenumber': [self.file_counter], 
        'particle_id': [self.original_pid], 
        'mc_z': [correct_hit.z], 
        'mc_r' : [correct_hit.r], 
        'pred_z': [a[0]], 
        'pred_r': [a[1]], 
        'reward': [reward] })
        row.to_csv(f, mode='a', header=None, index=None)
        num_layer_hits = self.original_particle.unique_layer_id.nunique()

        #if predicted point is above, there are no compatible hits and already predicted 7 hits or 
        #if (predicted_point_r > np.max(self.original_particle.r)) or (signal_done & (self.num_track_hits > 7)) :#or (done_ep) or ):
        #or (done==True) or (
        #    no_hit_in_layer &  self.num_track_hits > num_layer_hits): 
        if  self.num_track_hits > 6:
            done = True 
        else: 
            done = False 
            #self.episode_counter +=1 

        self._point = a
        #self.state = [self._point[0], self._point[1] , m, b] # self.dm, self.db]
        observation = self.state


       # self.average_reward = (self.average_reward + reward)/2
        

        return observation, reward, self.cor.hit_id, done


    def correct_hit(self): 
        try: 
            correct_hit = self.original_particle.iloc[self.num_track_hits]
        except: 
            print(len(self.original_particle), self.num_track_hits)
       
        return correct_hit 
       
        