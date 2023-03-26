import pandas as pd
import numpy as np 

from utils.geometry import calc_distance, normalise_data 
import json 
import csv 

f = open("utils/helperfiles/mapping_noMissing_noPtcut_1400.json")
#f = open("utils/helperfiles/mapping_noMissing_noPtcut__noDoubles_1000.json") 
# returns JSON object as 
# a dictionary*
mappings = json.load(f)


f =  open("utils/helperfiles/mapping_CMS_nocut_1000.json")
mappings_cms = json.load(f)


f = open("evaluation/comp_hits.csv", "w")
writer = csv.writer(f)
writer.writerow(["particle_id", "hit2_z", "hit2_r", "m", "b", "comp_hit_z", "comp_hit_r"])

class Find_Compatible_Hits_ModuleMap_Line_New:
    def __init__(self, hits): 
        self.hits = hits 
        self.done = False 
        self.prev_hit_buffer = hits.iloc[0]
        #self.prev_prev_buffer = hits.iloc[0]

    def _find_m_b(self, hit1, hit2):
        """Slope and intercept of straight line given two hits"""
        m = (hit2.r - hit1.r)/(hit2.z - hit1.z)
        b = hit2.r - m*hit2.z
        return m,b 

    def _find_module_compatible_hits(self, hit2, m): 
        """Find potential next hits that are compatible with the module mapping"""
                
        try:
            comp_mod = mappings[str(int(hit2.discrete_module_id))]
        except: 
            comp_mod = np.unique(self.hits.discrete_module_id.values)
        # only allow z that's larger either positive or negative than previous z 
       
        
        compz = self.hits[np.sign(m)*self.hits['z'] > np.sign(m)*(hit2.z + 0.1)]
        #print("len 1", len(compz))


        #compz = compz[compz['r'] > self.prev_hit_buffer.r]
        #print("len 2", len(compz))
        # ensure it predicts from the nest layer and r is also bigger 
        comp_hits = compz[((compz['discrete_module_id'].isin(comp_mod)) & 
                   (compz['r'] > (hit2.r +1))  & (compz['unique_layer_id']!=hit2.unique_layer_id))]# & 
        #print("len 3", len(comp_hits))


        if len(comp_hits) == 0: 
            #print("htats right i went here")
            if m > 0: 
                compz = self.hits[self.hits['z'] > hit2.z] 
            else: 
                compz= self.hits[self.hits['z'] < hit2.z]
            
            comp_hits = compz[(compz['discrete_module_id'].isin(comp_mod)) & 
                    (compz['r'] > hit2.r)]#& (compz['r'] != self.prev_hit_buffer.r)]# & 

        if len(comp_hits) ==0 : 
            print("mwhahha I went here loliz")
        # TODO: fix this 
            #comp_hits = pd.concat([hit2_df])
            hit2_df = pd.DataFrame([hit2])

            #comp_hits = hit2_df
            comp_hits = self.hits[self.hits['discrete_module_id'].isin(comp_mod)]
            if(len(comp_hits)==0):
                print("no hits in the comp mod even though comp is", comp_mod)
        #print("usin hit 2 as comp")
            self.done = True 


        #if len(comp_hits) ==0 : 
        #    print("hits the first")
        #    comp_hits = self.hits[(self.hits['discrete_module_id'].isin(comp_mod)) & (self.hits['r'] > hit2.r)]

         #   self.done = True 
        #else: 
        #    print("does not hit first")
        #if len(comp_hits) == 0: 
        #    comp_hits = self.hits[self.hits['r'] > hit2.r]


        self.prev_hit_buffer = hit2

        return comp_hits 
    
    def _find_line_compatible_hits(self, m,b, comp_hits, num_close):

        #print("at line comp, comps are", comp_hits[['z', 'r']])
        distances = calc_distance(m,b, comp_hits.z, comp_hits.r)

        # this is probably very slow for many hits 
        final = comp_hits.iloc[distances.argsort()[:num_close]]

        # for i in range(len(final)):
        #     comp_row = final.iloc[i]
        #     row = pd.DataFrame({'particle_id': [self.hit.particle_id], 
        #     'hit2_z': [self.hit.z],
        #     'hit2_r': [self.hit.r],  
        #     'm': [m],
        #     'b' : [b],
        #     'comp_hit_z': [comp_row.z], 
        #     'comp_hit_r': [comp_row.r]})
        #     row.to_csv(f, mode='a', header=None, index=None)
            
        return final 
    
    def hit_df(self, hit): 
        """Return the correct pandas row from the hit posistion"""
        hit_df = self.hits[(self.hits['z'] == hit[0]) & (self.hits['r'] == hit[1])]
        if hit_df.shape[0] != 1: 
            hit_df = hit_df.iloc[0] 
        return hit_df.squeeze() 

    def get_comp_hits(self, hit2, m, b, num_close): 
        #the state only includes the positions of the hits, get the full row 
        #self.prev_hit_buffer = hit2 
        mod_comp_hits = self._find_module_compatible_hits(hit2, m) 
        # if (hit2.particle_id == 4.053265090839839e+17) & (hit2.hit_id == 10038.0): 
        #     print("slope is ", m, mod_comp_hits.hit_id)
        comp_hits = self._find_line_compatible_hits(m, b, mod_comp_hits, num_close)

        #print(comp_hits)
        # randomly shuffle the hits! This is important, otherwise it learns to always select the closest one by having the same quality for all hits
        #comp_hits = comp_hits.sample(frac=1)
        #comp_hits = mod_comp_hits.iloc[:num_close]
        
        #comp_hits = comp_hits.sample(frac=1)
        self.prev_prev_buffer = hit2 
        return comp_hits, self.done 

    def get_comp_hits_notrow(self, hit2_z, hit2_r, m, b, num_close): 
        hit2 = self.hit_df([hit2_z, hit2_r])
        self.prev_hit_buffer = hit2 
        mod_comp_hits = self._find_module_compatible_hits(hit2, m) 
        comp_hits = self._find_line_compatible_hits(m, b, mod_comp_hits, num_close)
        self.prev_prev_buffer = hit2 
        return comp_hits, self.done 

    def get_all_hits(self): 
        return self.hits 


    def get_reward(self, hit2, correct_hit): 
         
        distance = np.sqrt((hit2.z-correct_hit.z)**2 + (hit2.r-correct_hit.r)**2)
        # end_hit = particle.iloc[-1] 
        reward = -distance
        #reward = -np.exp(distance)

        # if hit2.r == correct_hit.r: 
        #     reward = 10
        if hit2.hit_id == correct_hit.hit_id: 
            reward = 10
    #     # elif end_hit.hit_id == correct_hit.hit_id: 
    #     #     reward = 0
    #    # elif self.previous_state[1] == self.state[1]: 
    #     #    reward = -5
    #     else: 
    #         reward = -distance
        
        return reward 

    def get_reward_binary(self, comp_hits, correct_hit): 
        comp_hits = comp_hits.reset_index() 
        distances = np.sqrt((comp_hits.z-correct_hit.z)**2 + (comp_hits.r-correct_hit.r)**2)
        rewards = np.zeros(len(distances))        
        ix_right = comp_hits[comp_hits['hit_id'] == correct_hit.hit_id]
        rewards[ix_right.index] = 1 
        #print(comp_hits[['z', 'r']], correct_hit[['z', 'r']], rewards)
        return rewards 


    def get_rank_reward(self, comp_hits, correct_hit): 
        
        distances = np.sqrt((comp_hits.z-correct_hit.z)**2 + (comp_hits.r-correct_hit.r)**2)
        sorted = np.argsort(-distances.values)
        sorted2 = np.argsort(sorted)
        #print(distances.values, sorted)
        #print(distances.values, sorted2)

        return sorted2 

    # def get_reward(self, hit2, hit3):
    #     particle = self.hits[self.hits['particle_id'] == hit2.particle_id] 
    #     particle = particle.groupby('unique_layer_id').min().reset_index()

    #     op_wo_self = particle[particle['unique_layer_id']!=hit2.unique_layer_id]
    #     end_track = False 
    #     try: 
    #         correct_hit = op_wo_self[(op_wo_self['r'] > hit2[1])].iloc[0]
    #     except: 
    #         correct_hit = hit2
    #         end_track = True
    #         done = True

    #     distance = np.sqrt((hit3.z-correct_hit.z)**2 + (hit3.r-correct_hit.r)**2)
        
    #     if hit3.hit_id == correct_hit.hit_id: 
    #         reward = 10
    #     elif end_track: 
    #         reward = 0
    #    # elif self.previous_state[1] == self.state[1]: 
    #     #    reward = -5
    #     else: 
    #         reward = -distance
        
    #     return reward 
        

    # def get_correct_hit(self, hit2): 
    #     #particle = self.hits[self.hits['particle_id'] == hit2.particle_id] 
    #     # takes the smaller of the double hit 
        
    #     #particle = particle.groupby('unique_layer_id').min().reset_index()
    #     #correct_hit = particle.iloc[self.hit_counter]
        
        
    #     op_wo_self = particle[particle['unique_layer_id']!=hit2.unique_layer_id]
    #     op_wo_self= op_wo_self.sort_values(['r', 'z'])

        
    #     try: 
    #         correct_hit = op_wo_self[(op_wo_self['r'] > hit2.r)].iloc[0]
    #     except: 
    #         correct_hit = hit2

    #     #print("hit2", hit2[['z', 'r']], "\n particle", op_wo_self[['z', 'r']], "\n correct hit", correct_hit[['z',  'r']])
        
    #     return correct_hit

    def set_current_pid(self, pid): 
        self.current_pid = pid
    
    def set_counter(self, count): 
        self.hit_counter = count 

    def get_hit(self, hit_id): 
        return self.hits[self.hits['hit_id']==hit_id]

    def get_particle(self, pid): 
        # for debugging purposes 
        return self.hits[self.hits['particle_id']==pid]