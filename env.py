import pandas as pd 
import numpy as np 
import yaml 
import random 

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)




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
        self.file_number = 0 
        self.initial_event = pd.read_hdf(config['input_dir']+config['file_name']+str(self.file_number)+config['file_extension'])
        
        df = self.initial_event[self.initial_event['sim_pt'] > 2]
        self.event = df
        #self.event = pt_cut
        self.episode_counter = 0 
        self.record_partilce_ids = [] 
        self.record_r = [] 
        self.record_dr = [] 
        self.record_z = [] 
        self.record_dz = []
        self.record_new_r = []
        self.record_new_z = [] 
        self.previous_state = []
        self.reset_count = 0 
        self.write = 2 
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
        #self.event = self.event[self.event['sim_pt']>2]

    #ensuring the new hit picked can't be the same as the hit under consideration 
        other_hits = self.event[self.event['hit_id']!= self.state.hit_id]
        #find hits in the new x region (update to y and z)

        # this makes it faster to search for closest hit later on 
        compatible_hits = other_hits[(other_hits['r'] > (new_r -3)) & (other_hits['r'] < (new_r +3)) & (other_hits['z'] > (new_z - 5)) & (other_hits['z'] < (new_z + 5))]
        same_particle = compatible_hits[compatible_hits['particle_id']==self.original_pid]
        contains_same_track = len(same_particle) > 0 
        
        if contains_same_track: 
            distances = [] 
            #it's a big search, converting to list from pandas save an order of magnitude in time 
            rlist = same_particle.r.tolist()
            zlist = same_particle.z.tolist() 
            for ix in range(len(same_particle)): 
                row = same_particle.iloc[ix, ]
                distance = np.abs(new_r - rlist[ix]) + np.abs(new_z-zlist[ix])
                distances.append(distance)
            # the reward is 1/lenght of distance between projected place and the closest hit for that particle 
            reward = 1/min(distances)
            #reward = 1 
        else: 
            reward = -1
        
        if len(compatible_hits) == 0: 
            
            #return  np.array([self.state.r, self.state.z, self.previous_state.r, self.previous_state.z]), -1, True
            compatible_hits =  other_hits[(other_hits['r'] > (new_r - 10)) & (other_hits['r'] < (new_r + 10)) & (other_hits['z'] > (new_z - 100)) & (other_hits['z'] < (new_z + 100))]
            print("using full hits")
        # update the track to the hit with the closest r state 
        #closest_r_hit_idx = np.argmin(np.abs(compatible_hits.r - new_r))
        distances = [] 
        #it's a big search, converting to list from pandas save an order of magnitude in time 
        rlist = compatible_hits.r.tolist()
        zlist = compatible_hits.z.tolist() 
        for ix in range(len(compatible_hits)): 
            row = compatible_hits.iloc[ix, ]
            distance = np.abs(new_r - rlist[ix]) + np.abs(new_z-zlist[ix])
            distances.append(distance)
        new_hit = compatible_hits.iloc[np.argmin(distances), ]
      #  if new_hit.particle_id == self.original_pid: 
      #      reward =1
      #  else: 
      #      reward = -1 
        
 

      #  if new_hit.particle_id == self.state.particle_id: 
      #      reward = 1 
      #  else: 
      #      reward = -1 

        #new_hit = compatible_hits.iloc[closest_r_hit_idx]
        
        self.previous_state = self.state 
        self.state = new_hit 
        self.num_track_hits += 1 

        if self.num_track_hits > 4:
            done = True
        else: 
            done = False 
            self.episode_counter +=1 


        state = np.array([self.state.r, self.state.z, self.previous_state.r, self.previous_state.z])

        return state, reward, done
        
        
        
    def reset(self): 
        # start at random hit, but could start arbitrarly 
        #print("lenght event", len(self.event))
        #print("hit id", self.event.hit_id.values)
        random_hit_id = random.choice(self.event.hit_id.values)
        #print("hte hit id is ", random_hit_id)
        random_hit = self.event[self.event['hit_id']==random_hit_id]
        self.original_pid = random_hit.particle_id.values[0]
        np.savetxt('original_pid.csv', [self.original_pid], delimiter=',')

        #this turns it into a series instead of a df 
        self.state = random_hit.squeeze(axis=0)
        self.last_dr = None 
        self.num_track_hits = 0 
        self.reset_count += 1 
        if self.reset_count > 100: 
            self.file_number += 1 
            self.initial_event = pd.read_hdf(config['input_dir']+config['file_name']+str(self.file_number)+config['file_extension'])
            self.event = self.initial_event[self.initial_event['sim_pt'] > 2]
            self.reset_count = 0 
            print("jumping to file", self.file_number)

        return np.array([self.state.r, self.state.z, 0, 0])

