       #particle = particle[particle['hit_id']!=self.state.hit_id]
        #contains_same_track = len(same_particle) > 0 
        
        #if contains_same_track: 
        #distances = [] 
        #it's a big search, converting to list from pandas save an order of magnitude in time 
        #rlist = particle.r.tolist()
        #zlist = particle.z.tolist() 
        #for ix in range(len(particle)): 
        #    row = particle.iloc[ix, ]
        #    distance = np.sqrt((new_r - rlist[ix])**2 + (new_z-zlist[ix])**2)
        #    distances.append(distance)

            # the reward is 1/lenght of distance between projected place and the closest hit for that particle 
     #   if contains_same_track: 
     #       reward = 1/min(distances)
            #reward = 1 
     #   elif len(distances) > 0: 
      #      reward = -min(distances)
      #  else: 
      #      reward = -10 


        #if new_hit.particle_id == self.state.particle_id: 
        #    reward = 1 
        #else: 
        #    reward = -1 

       # isin = particle[particle['hit_id'].isin([self.state.hit_id, new_hit.hit_id])]
       # if len(isin) ==2: 
       #     index_hit1 = particle[particle['hit_id']==self.state.hit_id].index[0]
        #    index_hit2 = particle[particle['hit_id']==new_hit.hit_id].index[0]
            # if it's the very next hit, give a large reward 
           # if index_hit1 == (index_hit2 + 1): 
           #     reward = 10 
            # also reward if same particle 
            #else: 
            #    reward = 1 
       # else: 8uj
       #     reward = -1 
        

        #new_hit = compatible_hits.iloc[closest_r_hit_idx]
                 # self.write = 1 
        #self.event = self.event[self.event['sim_pt']>2]

    #ensuring the new hit picked can't be the same as the hit under consideration 
        #other_hits = self.event[self.event['hit_id']!= self.state.hit_id]
        #find hits in the new x region (update to y and z)

        # this makes it faster to search for closest hit later on 
        #compatible_hits = other_hits[(other_hits['r'] > (new_r -3)) & (other_hits['r'] < (new_r +3)) & (other_hits['z'] > (new_z - 5)) & (other_hits['z'] < (new_z + 5))]
        #same_particle = compatible_hits[compatible_hits['particle_id']==self.original_pid]
        
        #index_hit1 = particle[particle['hit_id']==self.state.hit_id].index[0]
        
        #print(index_hit1)
        #print(self.num_track_hits)

         
        #if len(compatible_hits) == 0: 
            
            #return  np.array([self.state.r, self.state.z, self.previous_state.r, self.previous_state.z]), -1, True
        #    compatible_hits =  other_hits[(other_hits['r'] > (new_r - 15)) & (other_hits['r'] < (new_r + 15)) & (other_hits['z'] > (new_z - 100)) & (other_hits['z'] < (new_z + 100))]

        #distances = [] 
        #it's a big search, converting to list from pandas save an order of magnitude in time 
        #rlist = compatible_hits.r.tolist()
        #zlist = compatible_hits.z.tolist() 
        #for ix in range(len(compatible_hits)): 
        #    row = compatible_hits.iloc[ix, ]
        #    distance = np.abs(new_r - rlist[ix]) + np.abs(new_z-zlist[ix])
        #    distances.append(distance)
        #new_hit = compatible_hits.iloc[np.argmin(distances), ]