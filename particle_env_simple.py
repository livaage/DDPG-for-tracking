"""Simple 2D environment containing a point and a goal location."""
import math

import akro
import numpy as np
import circle_fit as cf
import pandas as pd 
from garage import Environment, EnvSpec, EnvStep, StepType
import random 
from gym.spaces import Box
#from visualise_track import visualise 
from animate_particle import wrap 
from new_animate_particle import visualise 



def dip_angle(dr, dz): 
    if dz == 0: 
        dz = 0.01
    dip =  np.tan(dr/dz)

    if math.isnan(dip): 
        print(dr, dz)
    #print(dip)
    return dip

def azimuthal_angle(dx, dy): 
#print(dx, dy)
#x = np.tan(dy, dx)
    if dx ==0: 
        dx = 0.01
    angle = np.tan(dy/dx)
    return angle


def estimate_momentum(data): 
    xc,yc,r,_ = cf.least_squares_circle((data))
    #print(pt)
    pt = r*0.01*0.3*3.8  

    return pt 


#r = pd.read_csv('~/garage/src/garage/examples/tf/g_r.csv', header=None)
#z = pd.read_csv('~/garage/src/garage/examples/tf/g_z.csv', header=None)
#pids = pd.read_csv('~/garage/src/garage/examples/tf/g_pids.csv', header=None)

#i = np.where(pids.values.flatten()==-17737)

#my_r = r.values[i]
#my_z = z.values[i]
done_ani = False 

event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event0.h5')


class ParticleEnvSimple(Environment):
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
                 goal=np.array((1., 1.), dtype=np.float32),
                 arena_size=5.,
                 done_bonus=0.,
                 never_done=False,
                 max_episode_length=math.inf):
        goal = np.array(goal, dtype=np.float32)
        self._goal = goal
        self._done_bonus = done_bonus
        self._never_done = never_done
        self._arena_size = arena_size
        self._total_step_cnt = 0 
        self.new_count = 0 
        self.done_visual = False 
        self.file_counter = 0 
        self.event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event0.h5')
        self.average_reward = 0 
        self.hit_buffer = []


        assert ((goal >= -arena_size) & (goal <= arena_size)).all()

        self._step_cnt = None
        self._max_episode_length = max_episode_length
        self._visualize = False

        self._point = np.zeros_like(self._goal)
        self._task = {'goal': self._goal}
        self._observation_space = akro.Box(low=np.array([-266, 0, -100, -3]), high=np.array([266, 26, 100, 10]), dtype=np.float64)
        self._action_space = akro.Box(low=-4,
                                      high=20,
                                      shape=(2, ),
                                      dtype=np.float32)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)

        self.record_z = [] 
        self.record_r = []
        self.record_pid = []
        self.record_event_counter = [] 
        self.record_reward = [] 
        self.record_a0 = [] 
        self.record_a1 = [] 
        print("INIIIITIALLIIISED")

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
        
        if self._total_step_cnt%100 ==0: 
            self.file_counter += 1 
            self.event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event'+str(self.file_counter)+'.h5')
            print("jumping file")
        self.event = event[event['sim_pt'] > 2]
        #subset by the number of hits 
        nh = self.event.groupby('particle_id').agg('count').iloc[:,0]
        # only pick the pids that has a certain number of hits 
        self.event = self.event[self.event['particle_id'].isin(np.array(nh[nh > 7].index))]

        random_particle_id = random.choice(self.event.particle_id.values)
        self.particle = self.event[self.event['particle_id']==random_particle_id]
        #print(random_particle_id)
        self.original_pid = random_particle_id
        # This relies on an ordered df!  
        start_hit = self.particle.iloc[0,:]
        self._point = start_hit[['z', 'r']].values 
        next_hit = self.particle.iloc[1,:]
        self.hit_buffer = [] 
        self.hit_buffer.append([next_hit.x, next_hit.y])

        self.num_track_hits = 0 
        dist = np.linalg.norm(start_hit[['z', 'r']].values - next_hit[['z', 'r']].values)        
        #print(self._point, dist)
        self.state = start_hit.squeeze(axis=0) 
        dist = start_hit[['z', 'r']] - next_hit[['z', 'r']]
        dz = start_hit.z - next_hit.z
        dr = start_hit.r - next_hit.r
        dx = start_hit.x - next_hit.x
        dy = start_hit.y - next_hit.y


        self.record_z.append(start_hit.z)
        self.record_r.append(start_hit.r)
        self.record_z.append(next_hit.z)
        self.record_r.append(next_hit.r)
      

        #self.record_file.append(next_hit.r)
        #self.record_pid.append([self.original_pid, self.original_pid])
        self.record_pid.append(self.original_pid)
        self.record_pid.append(self.original_pid)



        #print(dr, dz, dx, dy)

        dip = dip_angle(dr, dz)
        phi = azimuthal_angle(dx, dy)

        
        observation = np.concatenate((self._point, [dz], [dr]))
        #print(observation)


        

        self._step_cnt = 0
        self.original_particle = self.event[self.event['particle_id']==self.original_pid].reset_index()

        return observation, dict(goal=self._goal)

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

        #print("i am stepping so new count is ", self.new_count)
        # enforce action space
        a = action.copy()  # NOTE: we MUST copy the action before modifying it
        a = np.clip(a, self.action_space.low, self.action_space.high)
        #print(a)

        #self._point = np.clip(self._point + a, -266)
        #                      266)
        predicted_point_z = np.clip(self._point[0] + a[0], -266, 266)
        predicted_point_r = np.clip(self._point[1] + a[1], 0, 27)

        #print(a[0], )
        predicted_point = [predicted_point_z, predicted_point_r]

        #print(predicted_point)
        
        if self._visualize:
            print(self.render('ascii'))

        other_hits = self.event[self.event['hit_id']!=self.state.hit_id]
        # it's a big search, converting to list from pandas save an order of magnitude in time,a also just search a small part of the df 
        zlist = other_hits.z.tolist()
        rlist = other_hits.r.tolist() 

        distances = np.sqrt((zlist-predicted_point[0])**2+(rlist - predicted_point[1])**2) 
        index = np.argmin(distances)
        
        new_hit = other_hits.iloc[index, ] 
        #distance_prev_hit = np.sqrt((self.state.r - new_hit.r)**2 + (self.state.z - new_hit.z)**2)
        distance_prev_hit = [self.state.z - new_hit.z, self.state.r - new_hit.r]
        mag_dist_prev_hit = np.sqrt(self.state.z-new_hit.z)**2 + (self.state.r-new_hit.r)**2
        self.previous_state = self.state
        self.state = new_hit 

        # this is dangerous - relies on ordered df! 
        next_hit = self.original_particle.loc[self.num_track_hits +1,: ]
        self.hit_buffer.append([new_hit.x, new_hit.y])

        #reward given based on how close this new hit was to the next hit in the df 
        distance = np.sqrt((new_hit.z - next_hit.z)**2 + (new_hit.r - next_hit.r)**2)
        #print(distance)
        reward = -distance
        #if (mag_dist_prev_hit < 1): 
        #    reward -=100

        self.num_track_hits += 1 


        dr = self.state.r - self.previous_state.r 
        dz = self.state.z - self.previous_state.z 
        dx = self.state.x - self.previous_state.x 
        dy = self.state.y - self.previous_state.y

        #print(dr, dz, dx, dy)
        
        dip = dip_angle(dr, dz)
        phi = azimuthal_angle(dx, dy)
        p = estimate_momentum(self.hit_buffer)

        self.record_pid.append(self.original_pid)
        self.record_z.append(new_hit.z)
        self.record_r.append(new_hit.r)
        self.record_event_counter.append(self.file_counter)
        self.record_reward.append(reward)
        self.record_a0.append(predicted_point_z)
        self.record_a1.append(predicted_point_r)

        self._step_cnt += 1
        self._total_step_cnt += 1
        #print(self._step_cnt)

        if (self._total_step_cnt > 10000) & (self._total_step_cnt < 10002): 
            print("I will now save the files ")
            np.savetxt('g_pids.csv', self.record_pid, delimiter=',')
            np.savetxt('g_z.csv', self.record_z, delimiter=',')
            np.savetxt('g_r.csv', self.record_r, delimiter=',')
            np.savetxt('g_filenumber.csv', self.record_event_counter, delimiter=',')
            np.savetxt('g_reward.csv', self.record_reward, delimiter=',')
            np.savetxt('g_a0.csv', self.record_a0, delimiter=',')
            np.savetxt('g_a1.csv', self.record_a1, delimiter=',')
           # pass 

        if (self._total_step_cnt ==20001) & (self.done_visual == False) : 
            print(self.done_visual, self._total_step_cnt)
            #self.my_visualise()
            self.done_visual =True 
            print("it shouldnt happen again")
            #x = 2
       
        if self.num_track_hits > 6:
            done = True 
        else: 
            done = False 
            #self.episode_counter +=1 

        self._point = [next_hit.z, next_hit.r]
        #distance_to_prev_hit = new_hit[['r', 'z']] - 

        observation = np.concatenate((self._point, [dz], [dr]))
        step_type = StepType.get_step_type(
            step_cnt=self._step_cnt,
            max_episode_length=self._max_episode_length,
            done=done)

        if step_type in (StepType.TERMINAL, StepType.TIMEOUT):
            self._step_cnt = None


        self.average_reward = (self.average_reward + reward)/2
        #if self._total_step_cnt%100==0: 
        #    print(self.average_reward)

        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=reward,
                       observation=observation,
                       env_info={
                           'task': self._task,
                           #'success': succ
                       },
                       step_type=step_type)

    def render(self, mode):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        Returns:
            str: the point and goal of environment.

        """
        return f'Point: {self._point}, Goal: {self._goal}'

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
        goals = np.random.uniform(-2, 2, size=(num_tasks, 2))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (dict[str, np.ndarray]): A task (a dictionary containing a
                single key, "goal", which should be a point in 2D space).

        """
        self._task = task
        self._goal = task['goal']

    def dump_summary(self):
        print("dr:   ", "\n dz:    " ) 

