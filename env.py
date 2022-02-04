import pandas as pd 
import numpy as np 
import yaml 
import random 

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


event = pd.read_hdf(config['input_dir'])


class TrackEnv(): 
    """ 
    ##Action space 
    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | dx     |-43.0 | 43.0|
    
    ## Observation Space
    The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.
    | Num | Observation      |   Min  |   Max |
    |-----|------------------|--------|-------|
    | 0   | x                | -25.4  | 25.4  |
    | 1   | y                | -25.4  | 25.4  |
    | 2   | z                | -266   | 266   |
    | 3   | particle_id      |  N/A   | N/A   |
    
    ## Rewards
    The reward is defined as:
    ```
    r = if close to hit 1 otherwise -1 
    ```
    ##Starting state
    Random hit 
    
    ##Arguments 
    """ 
    
    
    def __init__(self): 
        self.placeholder = 0 
    
    #completes one step based on the input action value 
    def step(self, dx): 

        new_x = self.state.x + dx 
        #checking that the new x is within the x limits
        new_x = np.clip(new_x, -25.4, 25.4)

        #find hits in the new x region (update to y and z)
        compatible_hits = event[(event['x'] > (new_x - 0.05)) & (event['x'] < (new_x + 0.05))]

        contains_same_track = len(compatible_hits[(compatible_hits['particle_id']==self.state.particle_id) 
                                              & (compatible_hits['hit_id']!=self.state.hit_id)]) > 0 
        if contains_same_track: 
            reward = 1
        else: 
            reward = -1 
        
        if len(compatible_hits) == 0: 
            print("no compatibles")
            return  np.array([self.state.x, self.state.y, self.state.z]), -1, True
        # update the track to the hit with the closest x state 
        closest_x_hit_idx = np.argmin(np.abs(compatible_hits.x - new_x))
        new_hit = compatible_hits.iloc[closest_x_hit_idx]
        
        self.state = new_hit 
        self.num_track_hits += 1 

        if self.num_track_hits > 10:
            done = True
        else: 
            done = False 

        state = np.array([self.state.x, self.state.y, self.state.z])

        return state, reward, done
        
        
        
    def reset(self): 
        # start at random hit, but could start arbitrarly 
        random_hit_id = random.choice(event.hit_id)
        random_hit = event[event['hit_id']==random_hit_id]
        #this turns it into a series instead of a df 
        self.state = random_hit.squeeze(axis=0)
        self.last_dx = None 
        self.num_track_hits = 0 
        return np.array([self.state.x, self.state.y, self.state.z])