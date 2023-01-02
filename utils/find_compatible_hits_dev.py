import pandas as pd
import numpy as np 

from utils.geometry import calc_distance 
import json 
import csv 

#f = open("/home/lhv14/mapping_noMissing_noPtcut_1400.json")
f = open("/home/lhv14/mapping_noMissing_noPtcut__noDoubles_1000.json") 
# returns JSON object as 
# a dictionary*
mappings = json.load(f)


f =  open("/home/lhv14/mapping_CMS_nocut_1000.json")
mappings_cms = json.load(f)


f = open("/home/lhv14/DDPG/evaluation/comp_hits.csv", "w")
writer = csv.writer(f)
writer.writerow(["particle_id", "hit2_z", "hit2_r", "m", "b", "comp_hit_z", "comp_hit_r"])

class Find_Compatible_Hits_ModuleMap_Line_New:
    def __init__(self, hits): 
        self.hits = hits 
        self.done = False 
        self.prev_hit_buffer = hits.iloc[0]
        self.prev_prev_buffer = hits.iloc[0]

    def _find_m_b(self, hit1, hit2):
        m = (hit2.r - hit1.r)/(hit2.z - hit1.z)
        b = hit2.r - m*hit2.z
        return m,b 

    def _find_module_compatible_hits(self, hit2, m): 
        
        self.hit = hit2 
        try: 
            comp_mod = mappings[str(int(hit2.discrete_module_id))]
            #comp_mod = mappings_cms[str(int(hit2.discrete_module_id))]
        except: 
            #print("that comp mod issue ", "hit 2 is", hit2, "mod id hit2", hit2.discrete_module_id)
            comp_mod = np.unique(self.hits.discrete_module_id.values)
            
        if m > 0: 
            compy = self.hits[self.hits['z'] > hit2.z + 0.01] 
        else: 
            compy = self.hits[self.hits['z'] < hit2.z - 0.01]

        comp_hits = compy[(compy['discrete_module_id'].isin(comp_mod)) & 
                    (compy['r'] > (hit2.r +0.1)) & (compy['r'] != self.prev_hit_buffer.r) & (compy['unique_layer_id']!=hit2.unique_layer_id)]# & 
                    #(np.abs(self.hits['z']) > np.abs(hit2.z))]
        if hit2.unique_layer_id == self.prev_hit_buffer.unique_layer_id == self.prev_prev_buffer.unique_layer_id: 
        #if hit2.unique_layer_id == self.prev_hit_buffer.unique_layer_id: 
            comp_hits = comp_hits[comp_hits['unique_layer_id']!=hit2.unique_layer_id]

        if len(comp_hits) == 0: 
            #print("htats right i went here")
            if m > 0: 
                compy = self.hits[self.hits['z'] > hit2.z] 
            else: 
                compy = self.hits[self.hits['z'] < hit2.z]
            
            comp_hits = compy[(compy['discrete_module_id'].isin(comp_mod)) & 
                    (compy['r'] > hit2.r) & (compy['r'] != self.prev_hit_buffer.r)]# & 

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

        return comp_hits 
    
    def _find_line_compatible_hits(self, m,b, comp_hits, num_close):
        distances = calc_distance(m,b, comp_hits.z, comp_hits.r)

        #if num_close == 0: 
        #    idx = np.argmin(distances)
        #    one_hit = pd.DataFrame([comp_hits.iloc[idx]]) 
        #    final = one_hit
            #final = pd.concat([one_hit, one_hit])
            #print(final)
        #elif len(comp_hits) > num_close: 
            #idx = np.argpartition(distances, num_close)
            #final = comp_hits.iloc[idx][:num_close]
        final = comp_hits.iloc[distances.argsort()[:num_close]]

        #else: 
        #    final = comp_hits 
        
        #print(final)
                              #prepare file to write output 
        #print("lenght of comp hits is", len(final))
        for i in range(len(final)):
            comp_row = final.iloc[i]
            row = pd.DataFrame({'particle_id': [self.hit.particle_id], 
            'hit2_z': [self.hit.z],
            'hit2_r': [self.hit.r],  
            'm': [m],
            'b' : [b],
            'comp_hit_z': [comp_row.z], 
            'comp_hit_r': [comp_row.r]})
            row.to_csv(f, mode='a', header=None, index=None)
            
        return final 
    
    def hit_df(self, hit): 
        hit_df = self.hits[(self.hits['z'] == hit[0]) & (self.hits['r'] == hit[1])]
        if hit_df.shape[0] != 1: 
            print("the hit df is", hit_df)
            hit_df = hit_df.iloc[0] 
        return hit_df.squeeze() 

    def get_comp_hits(self, hit2, m, b, num_close): 
        #the state only includes the positions of the hits, get the full row 
        self.prev_hit_buffer = hit2 
        mod_comp_hits = self._find_module_compatible_hits(hit2, m) 
        comp_hits = self._find_line_compatible_hits(m, b, mod_comp_hits, num_close)
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


        
