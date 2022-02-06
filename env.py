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
        self.episode_counter = 0 
        self.record_partilce_ids = [] 
        self.record_r = [] 
        self.record_dr = [] 
        self.record_z = [] 
        self.record_dz = []
        self.record_new_r = []
        self.record_new_z = [] 
        self.previous_state = []

        self.write = 0 
    #completes one step based on the input action value 
    def step(self, a): 
       
        dr = a[0]
        dz = a[1]

        
        new_r = self.state.r + dr 
        #checking that the new x is within the x limits
        new_r = np.clip(new_r, 0, 25.4)

        new_z = self.state.z + dz 
        new_z = np.clip(new_z, -266, 266)
        #if (self.episode_counter > 2800) & (self.episode_counter < 2850): 
            #print("episode counter", self.episode_counter)
        self.record_partilce_ids.append(self.state.particle_id)
        self.record_r.append(self.state.r)
        self.record_dr.append(dr)
        self.record_z.append(self.state.z)
        self.record_dz.append(dz)
        self.record_new_r.append(new_r)
        self.record_new_z.append(new_z)
        
    
        #if (self.episode_counter == 2850) & (self.write==2): 
        np.savetxt('pids.csv', self.record_partilce_ids, delimiter=',')
        np.savetxt('rs.csv', self.record_r, delimiter=',')
        np.savetxt('drs.csv', self.record_dr, delimiter=',')
        np.savetxt('zs.csv', self.record_z, delimiter=',')
        np.savetxt('dzs.csv', self.record_dz, delimiter=',')
        np.savetxt('new_r.csv', self.record_new_r, delimiter=',')
        np.savetxt('new_z.csv', self.record_new_z, delimiter=',')
           # self.write = 1 


        #find hits in the new x region (update to y and z)
        # this makes it faster to search for closest hit later on 
        compatible_hits = event[(event['r'] > (new_r - 0.1)) & (event['r'] < (new_r + 0.1)) & (event['z'] > (new_z - 1)) & (event['z'] < (new_z + 1))]
        
        #ensuring the new hit picked can't be the same as the hit under consideration 
        compatible_hits = compatible_hits[compatible_hits['hit_id']!= self.state.hit_id]

        contains_same_track = len(compatible_hits[compatible_hits['particle_id']==self.state.particle_id]) > 0 
        if contains_same_track: 
            reward = 10
        else: 
            reward = -1
        
        if len(compatible_hits) == 0: 
            print("no compatibles")
            return  np.array([self.state.r, self.state.z, self.previous_state.r, self.previous_state.z]), -1, True
        # update the track to the hit with the closest r state 
        #closest_r_hit_idx = np.argmin(np.abs(compatible_hits.r - new_r))
        distances = [] 
        for ix in range(len(compatible_hits)): 
            row = compatible_hits.iloc[ix, ]
            distance = np.abs(new_r - row.r) + np.abs(new_z-row.z)
            distances.append(distance)
        new_hit = compatible_hits.iloc[np.argmin(distances), ]


      #  if new_hit.particle_id == self.state.particle_id: 
      #      reward = 1 
      #  else: 
      #      reward = -1 

        #new_hit = compatible_hits.iloc[closest_r_hit_idx]
        
        self.previous_state = self.state 
        self.state = new_hit 
        self.num_track_hits += 1 

        if self.num_track_hits > 3:
            done = True
        else: 
            done = False 
            self.episode_counter +=1 


        state = np.array([self.state.r, self.state.z, self.previous_state.r, self.previous_state.z])

        return state, reward, done
        
        
        
    def reset(self): 
        # start at random hit, but could start arbitrarly 
        random_hit_id = random.choice(event.hit_id)
        random_hit = event[event['hit_id']==random_hit_id]
        #this turns it into a series instead of a df 
        self.state = random_hit.squeeze(axis=0)
        self.last_dr = None 
        self.num_track_hits = 0 
        return np.array([self.state.r, self.state.z, 0, 0])

