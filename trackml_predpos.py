"""Similar to the regular trackmlenv, but this time predicts position and not a correction!."""
from concurrent.futures import process
import math

import akro
from charset_normalizer import detect
import numpy as np
import circle_fit as cf
import pandas as pd 
#from garage import Environment, EnvSpec, EnvStep, StepType

import random 
from gym.spaces import Box
import csv
from sklearn.metrics import precision_recall_curve 
import trackml.dataset
import json
import yaml

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)



#detector = pd.read_csv('/home/lhv14/detector_fixed_endcaps.csv')

hor_vol = [8, 13, 17]
ver_vol = [7, 9, 12, 14, 16, 18]




md = pd.read_csv(config['md_file'],  header=[0,1], index_col=[0, 1,2])
md = md.reset_index()
md.rename(columns = {'level_1':'layer_id', 'level_2':'volume_id'}, inplace = True)

prefix = config['file_prefix'] 
f = open("garage_outputs.csv", "w")
writer = csv.writer(f)
writer.writerow(["filenumber", "particle_id", "mc_z", "mc_r", "pred_z", "pred_r", "action_z", "action_r", "reward"])

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

    def __init__(self,
                # goal=np.array((1., 1.), dtype=np.float32),
                 arena_size=5.,
                 done_bonus=0.,
                 never_done=False,
                 max_episode_length=500):
        #goal = np.array(goal, dtype=np.float32)
        #self._goal = goal
        self._done_bonus = done_bonus
        self._never_done = never_done
        self._arena_size = arena_size
        self._total_step_cnt = 0 
        self.new_count = 0 
        self.done_visual = False 
        self.file_counter = 10 

        self.average_reward = 0 
        self.hit_buffer = []
        self.dz_buffer = []
        self.dr_buffer = []



        self._step_cnt = None
        self._max_episode_length = max_episode_length
        self._visualize = False

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

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return [
            'ascii',
        ]

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
        self.event = pd.read_csv(prefix + str(1000) + '.csv')
        self.event = self.event.loc[self.event.groupby(['particle_id','volume_id', 'layer_id']).r.idxmin().values]
        #self.event =  self.event[self.event['pt'] > 2]
        self.event = self.event.sort_values(['r', 'z'])
        c = self.event.groupby(['particle_id']).count()

        self.event = self.event[self.event['particle_id'].isin(c[c['hit_id'] > 6].index)]
        hits = self.event.copy() 
        hits[['volume_id_2', 'layer_id_2', 'z_2', 'r_2' ]] = hits.groupby('particle_id')[['particle_id', 'volume_id', 'layer_id', 'z', 'r']].shift(-1)
        hitsnona = hits.dropna()
        hitsnona['dr'] = hits['r_2'] - hits['r']
        hitsnona['dz'] = hits['z_2'] - hits['z']

        #only considering the inner part of the detector/removes tracks with large jumps 
        forbiddens = np.unique(hitsnona[(np.abs(hitsnona['dr']) > 10) & (np.abs(hitsnona['dz']) > 10)].particle_id)

        self.event = self.event[~self.event['particle_id'].isin(forbiddens)]

        #seed = self.event.groupby('particle_id').head(2)

        # uncomment if using several files 
        #print(self.event)
        # if self._total_step_cnt%100 ==0: 
        #     self.file_counter += 1
            
        #     self.event = pd.read_csv(prefix + str(1000+self.file_counter) + '.csv')
        #     self.event = self.event.loc[self.event.groupby(['particle_id', 'layer_id']).r.idxmin().values]
        #     #self.event =  self.event[self.event['pt'] > 2]
        #     self.event = self.event.sort_values(['r', 'z'])
        #     c = self.event.groupby(['particle_id']).count()

        #     self.event = self.event[self.event['particle_id'].isin(c[c['hit_id'] > 6].index)]
         
            
        #     print("jumping file")
      

        random_particle_id = random.choice(self.event.particle_id.values)
        self.particle = self.event[self.event['particle_id']==random_particle_id]

        self.original_pid = random_particle_id
        start_hit = self.particle.iloc[0,:]
        next_hit = self.particle.iloc[1,:]#
        dr = next_hit.r - start_hit.r
        dz = np.clip(next_hit.z - start_hit.z, -15, 15)
        
   
        self._point = next_hit[['z', 'r']].values 
        self.dr_buffer = [dr]
        self.dz_buffer = [dz]

      
        self.num_track_hits = 1
        self.state = start_hit.squeeze(axis=0) 

        row = pd.DataFrame({'filenumber': [self.file_counter, self.file_counter], 
        'particle_id': [self.original_pid, self.original_pid], 
        'mc_z': [start_hit.z, next_hit.z],
        'mc_r' : [start_hit.r, next_hit.r],
        'pred_z': [np.nan, np.nan],
        'pred_r': [np.nan, np.nan],
        'action_z': [np.nan, np.nan],
        'action_r': [np.nan, np.nan], 
        'reward': [0, 0 ]})
        row.to_csv(f, mode='a', header=None, index=None)

        observation=np.append(self._point, start_hit[['z', 'r']].values)

        
        self.state = [next_hit.z, next_hit.r]

        self._step_cnt = 0
        self.original_particle = self.event[self.event['particle_id']==self.original_pid].reset_index()

        return observation

    def step(self, action):
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
        
        self.new_count += 1 

        # enforce action space
        #action = action*1000
        a = action.copy()  # NOTE: we MUST copy the action before modifying it

        
        a_clipped = np.clip(a, self.action_space.low, self.action_space.high)

        # already clipped, change 
        predicted_point_z = np.clip(a_clipped[0],  -300, 300)
        predicted_point_r = np.clip(a_clipped[1], 0, 120)


        predicted_point = [predicted_point_z, predicted_point_r]
       
        if self._visualize:
            print(self.render('ascii'))

        
        self.previous_state = self.state
        self.state = predicted_point


        next_index = self.num_track_hits + 1 
        if next_index > len(self.original_particle) -1: 
            next_index = len(self.original_particle) - 1
        next_hit = self.original_particle.loc[next_index,: ]

        
        distance = np.sqrt((predicted_point[0]-next_hit.z)**2 + (predicted_point[1]-next_hit.r)**2)
        reward = -distance
  
        self.num_track_hits += 1 
    

        dr = np.clip(self.state[1] - self.previous_state[1], 0, 10)
        dz = np.clip(self.state[0] - self.previous_state[0], -15, 15)
        
       
        self.dr_buffer.append(dr)
        self.dz_buffer.append(dz)
      
        self._step_cnt += 1
        self._total_step_cnt += 1
  

        row = pd.DataFrame({'filenumber': [self.file_counter], 
        'particle_id': [self.original_pid], 
        'mc_z': [next_hit.z], 
        'mc_r' : [next_hit.r], 
        'pred_z': [predicted_point_z], 
        'pred_r': [predicted_point_r], 
        'action_z': [a[0]], 
        'action_r': [a[1]], 
        'reward': [reward] })
        row.to_csv(f, mode='a', header=None, index=None)

        if self.num_track_hits > 5: 
        #if a[2] > 0.5:
            done = True 
        else: 
            done = False 
            #self.episode_counter +=1 

        self._point = predicted_point
        observation = np.append(self._point, self.previous_state)




        self.average_reward = (self.average_reward + reward)/2


        return observation, reward, done


    def render(self, mode):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        Returns:
            str: the point and goal of environment.

        """
        #return f'Point: {self._point}, Goal: {self._goal}'
        return self._point 

    def visualize(self):
        """Creates a visualization of the environment."""
        #self._visualize = True
        #print(self.render('ascii'))
        #visualise(self.state.r, )
        #visualise() 
        #wrap(self.event, r, z, pids, self.original_pid)
        print(self.original_pid)
        #print("i now visualise")

    def my_visualise(self): 
            print("now calling visualise")
            #wrap(self.event)
            #visualise(self.event, self.record_pid, self.record_r, self.record_z)
        
    def close(self):
        """Close the env."""

    # pylint: disable=no-self-use
    def sample_tasks(self, num_tasks):
        """Sample a list of `num_tasks` tasks.

        Args:
            num_tasks (int): Number of tasks to sample.

        Returns:
            list[dict[str, np.ndarray]]: A list of "tasks", where each task is
                a dictionary containing a single key, "goal", mapping to a
                point in 2D space.

        """
        #goals = np.random.uniform(-2, 2, size=(num_tasks, 2))
        #tasks = [{'goal': goal} for goal in goals]
        #return tasks
        return 0 


    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, np.ndarray]): A task (a dictionary containing a
                single key, "goal", which should be a point in 2D space).

        """
        #self._task = task
        #self._goal = task['goal']
        x = 10 

    def dump_summary(self):
        print("dr:   ", "\n dz:    " ) 


    # not used at the moment, ignore
    def find_intersects(self, hit1, hit2): 


        intersects = [] 

        m = (hit2.r - hit1.r)/(hit2.z - hit1.z)
        b = hit2.r - m*hit2.z

        #m = (hit2[1] - hit1[1])/(hit2[0] - hit1[0])
        #b = hit2[1] - m*hit2[0]

        #print(m, b)

        for vol in hor_vol: 
            sub = md[md['volume_id']==vol]
            
            for layer in sub.layer_id: 
                sub_layer = sub[sub['layer_id']==layer] 
                #print(sub_layer)
                z = (sub_layer['r_mean'].values -b)/m
                #print(sub_layer['cz']['max'].values, z)
                if (z < sub_layer['z_max'].values) & (z > sub_layer['z_min'].values):  
                    intersects.append([z[0],sub_layer['r_mean'].values[0], vol, layer])
                    #print(z[0])
                    
        for vol in ver_vol: 
            sub = md[md['volume_id']==vol]
            
            for layer in sub.layer_id: 
                sub_layer = sub[sub['layer_id']==layer] 
                #z = (b-sub_layer['cr']['mean'].values)/m
                r = m * sub_layer['z_mean'].values + b
                #print(sub_layer['cz']['max'].values, z)
                if (r < sub_layer['r_max'].values) & (r > sub_layer['r_min'].values):  
                    #intersects.append(np.array([sub_layer['cz']['mean'].values, r]).flatten())
                    intersects.append([sub_layer['z_mean'].values[0], r[0], vol, layer])

        # only consider hits in higher layers than seed
        intersects = np.array(intersects)
        if len(intersects) == 0 : 
            print("none found")
            return [] 
        #print(intersects)
        #intersects = intersects[intersects[:,1] > hit2.r + 1]
        #print("this is intersects", intersects)

        intersects = intersects[intersects[:, 1].argsort()]
        intersects = intersects[np.abs(intersects[:, 0]) > (np.abs(hit2.z) -2)]
        index_comp_hits = [] 
        close_hits = [] 

        #print("this is it", self.event[(self.event['volume_id']==14) & (self.event['layer_id']==12)])
        # check for compatible hits close to intersection area 
        for i in range(intersects.shape[0]): 
            interrow = intersects[i, ]
            h_po = self.event[(self.event['volume_id']==interrow[2]) & (self.event['layer_id'] == interrow[3])]
            #print(h_po, "vol id ", interrow[2], "layer id", interrow[3])
            #h_po = h_po[(h_po['z'] < interrow[0] + 2) & (h_po['z'] > interrow[0] - 2) &
            #    (h_po['r'] < interrow[1] + 1.5) & (h_po['r'] > interrow[1] - 1.5)]

            distances = np.sqrt((h_po.z - interrow[0])**2 + (h_po.r - interrow[1])**2)
            try: 
                index_closest = np.argmin(distances)
                close_hits.append(h_po.iloc[index_closest, ][['z', 'r']].values)
            except: 
                close_hits.append([interrow[0], interrow[1]])
            
            #if len(h_po) > 0: 
            #    index_comp_hits.append(i)

        #print(index_comp_hits)
        #intersects = intersects[index_comp_hits]
        #print("these are close hits", close_hits)
        #intersects = intersects[intersects[:, 1].argsort()]
        #intersects = intersects[:, 0:2]
        #intersects = intersects[np.abs(intersects[:, 0]) > (np.abs(hit2.z) -2)]
        #print(intersects)
       # intersects = np.array(intersects)

        return intersects
