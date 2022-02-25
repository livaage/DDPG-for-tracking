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

        self.episode_counter = 0 
        self.record_partilce_ids = [] 
        self.record_r = [] 
        self.record_dr = [] 
        self.record_z = [] 
        self.record_dz = []
        self.record_new_r = []
        self.record_new_z = [] 
        self.record_next_hit_r = []
        self.record_next_hit_z = [] 
        self.previous_state = []
        self.record_original_pid = [] 
        self.reset_count = 0 
        self.write = 2 
    #completes one step based on the input action value 
    def step(self, a): 
       
        dr = a[0]
        dz = a[1]
        
        new_r = self.state.r + dr
        #checking that the new x is within the x limits
        
        #this should be superflous 
        new_r = np.clip(new_r, 0, 26)

        new_z = self.state.z + dz 
        new_z = np.clip(new_z, -265, 265)
        
        #particle = particle[particle['hit_id']!=self.state.hit_id]
        #contains_same_track = len(same_particle) > 0 
        
        # find closest hit in the event to the predicted position 
        # remove the original hit from the df 
        other_hits = self.event[self.event['hit_id']!=self.state.hit_id]
        # it's a big search, converting to list from pandas save an order of magnitude in time,a also just search a small part of the df 
        rlist = other_hits.r.tolist()
        zlist = other_hits.z.tolist() 
        distances = np.sqrt((rlist-new_r)**2+(zlist - new_z)**2) 
        index = np.argmin(distances)
        new_hit = other_hits.iloc[index, ] 


        # this is dangerous - relies on ordered df! 
        next_hit = self.original_particle.loc[self.num_track_hits +1,: ]
        #reward given based on how close the hit was 
        distance = np.sqrt((new_hit.r - next_hit.r)**2 + (new_hit.z - next_hit.z)**2)
        reward = -distance

        
 

        self.previous_state = self.state 
        self.state = new_hit 
        self.num_track_hits += 1 
        phi = self.calculate_phi(new_hit.hit_id)

        dr = self.state.r - self.previous_state.r 
        dz = self.state.z - self.previous_state.z 

        if self.num_track_hits > 6:
            done = True 
        else: 
            done = False 
            self.episode_counter +=1 



        state = np.array([self.state.r, dr, self.state.z, dz])


        # stores and writes result to file         
        if (self.episode_counter > 4800) & (self.episode_counter < 4850): 
            #print("episode counter", self.episode_counter)
            self.record_partilce_ids.append(self.state.particle_id)
            self.record_r.append(self.state.r)
            #self.record_dr.append(dr)
            self.record_z.append(self.state.z)
            #self.record_dz.append(dz)
            self.record_new_r.append(new_r)
            self.record_new_z.append(new_z)

            self.record_next_hit_r.append(next_hit.r)
            self.record_next_hit_z.append(next_hit.z)

            #self.record_new_z.append(new_z)
            self.record_original_pid.append(self.original_pid)

    
        if (self.episode_counter == 5000) & (self.write==2): 
            np.savetxt('pids.csv', self.record_partilce_ids, delimiter=',')
            np.savetxt('rs.csv', self.record_r, delimiter=',')
            #np.savetxt('drs.csv', self.record_dr, delimiter=',')
            np.savetxt('zs.csv', self.record_z, delimiter=',')
            #np.savetxt('dzs.csv', self.record_dz, delimiter=',')
            np.savetxt('new_r.csv', self.record_new_r, delimiter=',')
            np.savetxt('new_z.csv', self.record_new_z, delimiter=',')
            np.savetxt('next_hit_r.csv', self.record_next_hit_r, delimiter=',')
            np.savetxt('next_hit_z.csv', self.record_next_hit_z, delimiter=',')
            np.savetxt('original_pid.csv', self.record_original_pid, delimiter=',')

        return state, reward, done
        
        
        
    def reset(self): 

        self.initial_event = pd.read_hdf(config['input_dir']+config['file_name']+str(self.file_number)+config['file_extension'])
        # cut on pt 
        self.event = self.initial_event[self.initial_event['sim_pt'] > 2]
        #subset by the number of hits 
        nh = self.event.groupby('particle_id').agg('count').iloc[:,0]
        # only pick the pids that has a certain number of hits 
        self.event = self.event[self.event['particle_id'].isin(np.array(nh[nh > 7].index))]

        # assume each file has about 100 tracks that has pt > 2 and >7 hits. If reusing tracks, that's fine as well 
        if self.reset_count > 100: 
            self.file_number += 1 
            self.initial_event = pd.read_hdf(config['input_dir']+config['file_name']+str(self.file_number)+config['file_extension'])
            self.event = self.initial_event[self.initial_event['sim_pt'] > 2]
            #subset by the number of hits 
            nh = self.event.groupby('particle_id').agg('count').iloc[:,0]
            # only pick the pids that has a certain number of hits 
            self.event = self.event[self.event['particle_id'].isin(np.array(nh[nh > 7].index))]
            self.reset_count = 0 
            print("jumping to file", self.file_number)


        random_particle_id = random.choice(self.event.particle_id.values)
        self.particle = self.event[self.event['particle_id']==random_particle_id]
        self.original_pid = random_particle_id
        # This relies on an ordered df!  
        start_hit = self.particle.iloc[0:,]

        
        self.index1 = 0

        # start with a seed so one has dr and dz 
        hit2  = self.particle.iloc[self.index1+1,:]
        phi2 = self.calculate_phi(hit2.hit_id)
        #del_phi = self.calculate_phi(random_hit.hit_id.value[0]) - phi2 
        #print(hit2.z)
        dz = start_hit.z.values[0] - hit2.z
        dr = start_hit.r.values[0] - hit2.r

       

        #this turns it into a series instead of a df 
        self.state = hit2.squeeze(axis=0)

        self.last_dr = None 
        self.num_track_hits = 2 
        self.reset_count += 1 
        self.original_particle = self.event[self.event['particle_id']==self.original_pid].reset_index()
        # 
 

        return np.array([self.state.r, dr, self.state.z, dz])

    def calculate_phi(self, hit_id): 
        h = self.event[self.event['hit_id']==hit_id]
        angle = np.arctan(h.y.values[0]/np.abs(h.x.values[0]))

        if h.x.values[0] < 0: 
            angle = np.pi - angle 
        return angle 
