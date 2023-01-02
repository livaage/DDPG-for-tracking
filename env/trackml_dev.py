"""Similar to the regular trackmlenv, but this time predicts position and not a correction!."""
from concurrent.futures import process
import math

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
from utils.geometry import find_n_closest_hits


f = open("evaluation/garage_outputs.csv", "w")
writer = csv.writer(f)
writer.writerow(["filenumber", "particle_id", "mc_z", "mc_r", "pred_z", "pred_r", "reward"])

class TrackMLPredposEnv():
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

    def __init__(self, hits):
        self._total_step_cnt = 0 
        self._step_cnt = None
        self.event = hits 
        # hit and previous hit is observation 
        self._observation_space = akro.Box(low=np.array([-300, 0, -300, 0]), high=np.array([300, 120, 300, 120]), dtype=np.float64)
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
        self.number_close_hits = 3


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
      

        random_particle_id = random.choice(np.unique(self.event.particle_id.values)[:2])
        #first_particle_id = self.event.particle_id.iloc[0]
        
        #self.particle = self.event[self.event['particle_id']==first_particle_id]
        
        self.particle = self.event[self.event['particle_id']==random_particle_id]
        

        self.original_pid = random_particle_id
        #self.original_pid = first_particle_id

    
        start_hit = self.particle.iloc[0,:]
        next_hit = self.particle.iloc[1,:]#
   
        self._point = next_hit[['z', 'r']].values 

      
        self.num_track_hits = 1

        #fix
        self.file_counter = 0 

        #prepare file to write output 
        row = pd.DataFrame({'filenumber': [self.file_counter, self.file_counter], 
        'particle_id': [self.original_pid, self.original_pid], 
        'mc_z': [start_hit.z, next_hit.z],
        'mc_r' : [start_hit.r, next_hit.r],
        'reward': [0, 0 ]})
        row.to_csv(f, mode='a', header=None, index=None)

        
        self.state = [next_hit.z, next_hit.r]

        self._step_cnt = 0
        self.original_particle = self.event[self.event['particle_id']==self.original_pid].reset_index()

        closest_hits = find_n_closest_hits(start_hit.z, start_hit.r, self.event, self.number_close_hits)

        observation=np.append(self._point, start_hit[['z', 'r']].values)

        #print("closest hits AAAAAAAAAAAAAAAAAAAAAAAAAAA", closest_hits)
        return observation, closest_hits

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
        if self._step_cnt is None:
            raise RuntimeError('reset() must be called before step()!')
        
        #action = action*1000
        a = action.copy()  # NOTE: we MUST copy the action before modifying it

        
        a_clipped = np.clip(a, self._action_space.low, self._action_space.high)

        # already clipped, change 
        predicted_point_z = np.clip(a_clipped[0],  -300, 300)
        predicted_point_r = np.clip(a_clipped[1], 0, 120)


        predicted_point = [a_clipped[0], a_clipped[1]]
        #print("predicted point", predicted_point)
       
        
        self.previous_state = self.state
        self.state = predicted_point

        hit_layer = np.unique(self.event[(self.event['z'] == predicted_point_z) & (self.event['r'] == predicted_point_r)].unique_layer_id)[0]
        #print(hit_layer)
        correct_hit_inthatlayer = self.original_particle[self.original_particle['unique_layer_id']==hit_layer]
        no_hit_in_layer = False

        if len(correct_hit_inthatlayer) > 1: 
            distance = np.sqrt((predicted_point[0]-correct_hit_inthatlayer.z.values)**2 + (predicted_point[1]-correct_hit_inthatlayer.r.values)**2)
            next_hit = correct_hit_inthatlayer.iloc[np.argmin(distance)].squeeze()
            #print("in first ", next_hit.z)
            distance = np.min(distance)
            #print(distance)
            
        elif len(correct_hit_inthatlayer) ==1: 
            next_hit = correct_hit_inthatlayer.squeeze() 
            distance = np.sqrt((predicted_point[0]-next_hit.z)**2 + (predicted_point[1]-next_hit.r)**2)
            #print(distance)
        else: 
            # no hits in that layer
            next_index = self.num_track_hits + 1 
            if next_index > len(self.original_particle) -1: 
                next_index = len(self.original_particle) - 1
            next_hit = self.original_particle.loc[next_index,: ]
            no_hit_in_layer = True
            distance  = 0 
            #print(distance)

        # next_index = self.num_track_hits + 1 
        # if next_index > len(self.original_particle) -1: 
        #     next_index = len(self.original_particle) - 1
        # next_hit = self.original_particle.loc[next_index,: ]

        
        #distance = np.min(np.sqrt((predicted_point[0]-self.original_particle.z)**2 + (predicted_point[1]-self.original_particle.r)**2))
        reward = -distance

        gone_beyond = False
        if predicted_point_r > np.max(self.original_particle.r) + 10: 
            gone_beyond = True
            reward = 0
    

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
        'pred_z': [predicted_point_z], 
        'pred_r': [predicted_point_r], 
        'reward': [reward] })
        row.to_csv(f, mode='a', header=None, index=None)
        num_layer_hits = self.original_particle.unique_layer_id.nunique()

        #if (self.num_track_hits > 7) or 
        if (predicted_point_r > np.max(self.original_particle.r)) or ((done==True) & (self.num_track_hits > 7)) or (
            no_hit_in_layer &  self.num_track_hits > num_layer_hits) or (gone_beyond): 

        #if a[2] > 0.5:
            done = True 
        else: 
            done = False 
            #self.episode_counter +=1 

        self._point = predicted_point
        observation = np.append(self._point, self.previous_state)




       # self.average_reward = (self.average_reward + reward)/2


        return observation, reward, done


    def close(self):
        """Close the env."""

    # pylint: disable=no-self-use
    def dump_summary(self):
        print("dr:   ", "\n dz:    " ) 
