import pandas as pd
import numpy as np 
from utils.geometry import calc_distance 
import json 


f = open("/home/lhv14/mapping_noMissing_noPtcut_1400.json")
  
# returns JSON object as 
# a dictionary*
mappings = json.load(f)

class Find_Compatible_Hits_ModuleMap_Line:
    def __init__(self, hits): 
        self.hits = hits 
        self.done = False 

    def _find_m_b(self, hit1, hit2):
        m = (hit2.r - hit1.r)/(hit2.z - hit1.z)
        b = hit2.r - m*hit2.z
        return m,b 

    def _find_module_compatible_hits(self, hit1, hit2): 
        #try: 
        comp_mod = mappings[str(int(hit2.discrete_module_id))]
        #except: 
        #    print("that comp mod issue ", "hit 2 is", hit2, "mod id hit2", hit2.discrete_module_id)
            
        if hit2.z - hit1.z > 0: 
            compy = self.hits[self.hits['z'] > hit2.z + 0.1] 
        else: 
            compy = self.hits[self.hits['z'] < hit2.z - 0.1]

        comp_hits = compy[(compy['discrete_module_id'].isin(comp_mod)) & 
                    (compy['r'] > hit2.r)]# & 
                    #(np.abs(self.hits['z']) > np.abs(hit2.z))]
       # if hit1.unique_layer_id == hit2.unique_layer_id: 
        comp_hits = comp_hits[comp_hits['unique_layer_id']!=hit2.unique_layer_id]
        if len(comp_hits) == 0: 
            hit2_df = pd.DataFrame([hit2])
            # TODO: fix this 
            #comp_hits = pd.concat([hit2_df])
            comp_hits = hit2_df
            #print("usin hit 2 as comp")
            self.done = True 

        return comp_hits 
    
    def _find_line_compatible_hits(self, m,b, comp_hits, num_close):
        distances = calc_distance(m,b, comp_hits.z, comp_hits.r)

        if num_close == 0: 
            idx = np.argmin(distances)
            one_hit = pd.DataFrame([comp_hits.iloc[idx]]) 
            final = pd.concat([one_hit, one_hit])
            #print(final)
        elif len(comp_hits) > num_close: 
            idx = np.argpartition(distances, num_close)
            final = comp_hits.iloc[idx.values][:num_close]

        else: 
            final = comp_hits 
        
        #print(final)

        return final 


    def get_comp_hits(self, hit1, hit2, num_close): 
        #the state only includes the positions of the hits, get the full row 
        hit1 = self.hits[(self.hits['z'] == hit1[0]) & (self.hits['r'] == hit1[1])].squeeze()
        hit2 = self.hits[(self.hits['z'] == hit2[0]) & (self.hits['r'] == hit2[1])].squeeze() 
        if len(self.hits[(self.hits['z'] == hit2[0]) & (self.hits['r'] == hit2[1])]) > 0: 
            hit2 = hit2.iloc[0]

        m, b = self._find_m_b(hit1, hit2) 
        mod_comp_hits = self._find_module_compatible_hits(hit1, hit2) 
        comp_hits = self._find_line_compatible_hits(m, b, mod_comp_hits, num_close)
        return comp_hits, self.done 
        
