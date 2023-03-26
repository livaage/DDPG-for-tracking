import numpy as np
import pandas as pd 
import trackml.dataset 
from sklearn import preprocessing
from configs import BASEPATH, BASE_DIR

#print("base dir is ", BASE_DIR)

unique_layer_id_mapping = pd.read_csv(BASE_DIR+'/utils/helperfiles/unique_layer_id_mapping.csv') 
allowed_layer_connections = pd.read_csv(BASE_DIR+'/utils/helperfiles/allowed_layer_connections.csv')
remap_modules_dic = ''
with open(BASE_DIR+'/utils/helperfiles/remap_modules.txt','r') as f:
         for i in f.readlines():
            remap_modules_dic=i #string
remap_modules_dic = eval(remap_modules_dic) # this is orignal dict with instace dict


prefix = '/home/lhv14/exatrkx/Tracking-ML-Exa.TrkX/alldata/train_2/event00000' 


#cms stuff 

allowed_layer_connections_cms = pd.read_csv(BASE_DIR+'/utils/helperfiles/allowed_layer_connections_cms.csv')
allowed_layer_connections_cms = allowed_layer_connections.rename(columns={'allowed_layer_id':'unique_layer_id'})

remap = {1:4, 2:0, 3:16, 5:28, 4:34, 6:39}

le = preprocessing.LabelEncoder()

def create_unique_label_background_tracks(x, y):
    x = -le.transform(y)
    return x

#prefix_cms = '/home/lhv14/iterativegnn/graph_construction/gnnfiles/ntuple_PU200_numEvent1000/ntuple_PU200_event'
prefix_cms = '/home/lhv14/ntuple/PU200/ntuple_PU200_'
class DataLoader:
    """Preprocesses and loads data. Returns processed hits"""  

    def __init__(self): 
        self.hits = pd.DataFrame({})
        self.all_hits = pd.DataFrame({})

    
    def _load_file_trackml(self, filenumber): 
        #self.hits = pd.read_csv(prefix+str(2820)+'.csv') 
        hits, particles, truth = trackml.dataset.load_event(
        prefix+str(2820+filenumber), parts=['hits', 'particles', 'truth'])


        hits['r'] = np.sqrt(hits.x**2 + hits.y**2)/10

        particles['pt'] = np.sqrt(particles.px**2 + particles.py**2)

        truth = (truth[['hit_id', 'particle_id']]
                    .merge(particles[['particle_id', 'pt', 'nhits', 'q']], on='particle_id'))

        phi = np.arctan2(hits.y, hits.x)

        # Select the data columns we need
        hits = (hits[['hit_id', 'x', 'y', 'z', 'r', 'layer_id', 'volume_id', 'module_id']]
                .assign(phi=phi)
                .merge(truth[['hit_id', 'particle_id', 'pt', 'nhits', 'q']], on='hit_id'))

        # Remove duplicate hits

        hits['z'] = hits['z']/10
        hits['x'] = hits['x']/10
        hits['y'] = hits['y']/10

        hits = hits.sort_values(['r', 'z'])
        
        #hits['new_module_id'] = hits['volume_id']*100000000+ hits['layer_id']*100000 + hits['module_id']
        mods = hits.set_index(['volume_id', 'layer_id', 'module_id']).index

        hits['discrete_module_id'] = [remap_modules_dic[val] for val in mods]
        hits = hits.merge(unique_layer_id_mapping, on=['volume_id', 'layer_id'], how='left')
        # removes two hits in one layer
        #hits = hits.groupby(['particle_id', 'unique_layer_id']).min().reset_index()
        s = hits.groupby('particle_id')['particle_id'].cumcount() 
        hits['hit_number'] = s


        #hits = hits.merge(lids, on=['volume_id', 'layer_id'], how='left')
        hits = hits.sort_values(['r', 'z'])
        
        #hits = hits.loc[hits.groupby(['particle_id', 'layer_id']).r.idxmin().values]
        #hits = hits.sort_values('r')


        self.all_hits = hits
        self.hits = hits 

    def _load_file_cms(self, filenumber): 
                #pre-processing 
        #hits = pd.read_hdf(prefix_cms+str(filenumber)+'.h5')
        hits = pd.read_hdf('/home/lhv14/ntuple/PU200/ntuple_PU200_' +str(filenumber)+'.h5')
        #hits = pd.read_hdf('/home/lhv14/ntuple/noPU/ntuple_noPU_' +str(filenumber)+'.h5')
        hits = hits.rename(columns={'gen_pt':'pt', 'gen_phi':'phi', 'gen_eta':'eta'})
        hits = hits.sort_values(['r', 'z'])
        hits['unique_layer_id'] = [remap[y] + x for x,y in hits[['layer_id', 'volume_id']].values]
        hits['discrete_module_id'] = [int(str(row[0]) + str(row[1]) + str(row[2])) for row in hits[['volume_id', 'unique_layer_id','module_id']].values] 
        #create new id for background hits - necessary because of how ntuples was written 
        background = hits[hits['particle_id']==-1]
        tracks = hits[hits['particle_id']!=-1]
        le.fit(background['sim_pt'].unique())
        background['particle_id'] = create_unique_label_background_tracks(background['particle_id'], background['sim_pt'])
        hits = tracks.append(background)
        s = hits.groupby('particle_id')['particle_id'].cumcount() 
        hits['hit_number'] = s
        # make a copy to make sure we have all the hits before subsetting 
        all_hits = hits 
        
        #start subsetting the ones we want to train on 
        
        # tracks with at least seven hits 
        
        # if a track has more than fifty hits, we don't want to train on it, likely very strange
        count = hits['particle_id'].value_counts()
        not_allowed = count[count > 50].index
        hits = hits[~hits['particle_id'].isin(not_allowed)]
        
        g = hits.groupby(['particle_id', 'unique_layer_id']).count()
        # you can't have more than 7 hits in a layer without being fake
        forbidden_pids = g[g['run']>7].index.get_level_values(0)
        hits = hits[~hits['particle_id'].isin(forbidden_pids)]
        
        hits = hits.sort_values(['r', 'z'])
        hits[['next_mod_id', 'connecting_layer_id']] = hits.groupby('particle_id')['discrete_module_id', 'unique_layer_id'].shift(-1)
        hits = hits.dropna()
        


        # remove tracks that have missing hits 
        allowed_hits = hits.merge(allowed_layer_connections_cms, on=['unique_layer_id', 'connecting_layer_id'], how='outer', indicator=True)
        hits = allowed_hits[allowed_hits['_merge']=='both']

        #z0 = hits.groubpy('particle_id')['z'].min() 
        
        c = hits.groupby(['particle_id']).count()
        hits = hits[hits['particle_id'].isin(c[c['hit_id'] > 8].index)]


        hits = hits.sort_values(['r', 'z'])

        self.hits = hits
        self.all_hits = all_hits 

    def _pt_cut(self, pt_min): 
        self.hits = self.hits[self.hits['pt'] > pt_min] 

    def _filter_by_num_track_hits(self, min_hits): 
        no_doubles = self.hits.groupby(['particle_id', 'unique_layer_id']).min().reset_index()

        c = no_doubles.groupby(['particle_id']).count()
        self.hits = self.hits[self.hits['particle_id'].isin(c[c['hit_id'] > min_hits].index)]

    def _assign_unique_layer_id(self): 
        self.hits = self.hits.merge(unique_layer_id_mapping, on=['volume_id', 'layer_id'], how='left')
        self.hits = self.hits.sort_values(['r', 'z'])
    
    def _get_layer_module_connections(self): 
        self.hits = self.hits.sort_values(['r', 'z'])
        self.hits[['next_mod_id', 'connecting_layer_id']] = self.hits.groupby('particle_id')['discrete_module_id', 'unique_layer_id'].shift(-1)
        self.hits = self.hits.dropna()

    def _filter_out_missing_hits(self): 
        allowed_hits = self.hits.merge(allowed_layer_connections, on=['unique_layer_id', 'connecting_layer_id'], how='outer', indicator=True)
        forbidden_pids = np.unique(allowed_hits[allowed_hits['_merge']=='left_only'].particle_id)
        self.hits = self.hits[~self.hits.isin(forbidden_pids)]

    def _filterout_na(self): 
        self.hits = self.hits.dropna() 

    def _sort(self): 
        self.hits= self.hits.sort_values(['r', 'z'])

    def _remove_bad_double_hits(self): 
        self.hits['z_2'] = self.hits.groupby('particle_id')['z'].shift(-1)
        self.hits['diff_z'] = self.hits['z_2'] - self.hits['z']
        z_0 = self.hits.groupby('particle_id').min().z
        right = z_0[z_0 > 0]
        left = z_0[z_0 < 0]
        right_particles = self.hits[self.hits['particle_id'].isin(right.index)].reset_index()
        left_particles = self.hits[self.hits['particle_id'].isin(left.index)].reset_index()
        ix_right = np.where(right_particles['diff_z'] < 0)[0]
        right_particles = right_particles.drop(ix_right + 1, axis = 0 )

        ix_left = np.where(left_particles['diff_z'] > 0)[0]
        left_particles = left_particles.drop(ix_left + 1, axis = 0 )
        f = right_particles.append(left_particles)
        return f

    def _sub_volume(self): 
        
        allowed_volume_id = [8, 13, 17] 
        g = self.hits.groupby(['particle_id', 'volume_id']).count().reset_index()

        forbidden_pids = g[~g['volume_id'].isin(allowed_volume_id)].particle_id.unique()

        self.hits = self.hits[~self.hits['particle_id'].isin(forbidden_pids)]




    def load_data_trackml(self, file_number): 
        self._load_file_trackml(file_number)
        #self._load_file_cms(file_number)
        self._remove_bad_double_hits()
        self._filter_by_num_track_hits(8) 
        #self._assign_unique_layer_id() 
        self._get_layer_module_connections()
        self._filter_out_missing_hits()
        #print(self.hits)
        #self._pt_cut(2)

        self._filterout_na() 
        self._pt_cut(2)
        self._sort() 
        #print(np.unique(self.hits['particle_id']))
        #self._sub_volume() 

        #return self.all_hits, np.unique(self.hits['particle_id'])10
        good_pids = self.hits.particle_id.unique()[:10]
        
        #print("goood pids", good_pids)
        #good_pids = [5.719682852312842e+17] #, 3.2876593389397606e+17]
        
        #self.hits = self.hits[self.hits['particle_id'].isin(good_pids)]

        #removable_pids = [pid for pid in np.unique(self.all_hits.particle_id) if pid not in np.unique(self.hits.particle_id)]

        #pids_to_remove = removable_pids[:round(len(removable_pids)/2)]
        #self.all_hits = self.all_hits[~self.all_hits['particle_id'].isin(pids_to_remove)]

        #allowed = np.unique(self.hits.particle_id)
        #self.hits = self.hits[self.hits['particle_id'].isin(good_pids)]
        #test_allowed_pids = allowed[:round(len(allowed)*0.001)]
        #test_allowed_pids = allowed[:2]
        #print(len( np.unique(self.hits.particle_id)))
        #self.all_hits = self.all_hits[self.all_hits['particle_id'].isin(test_allowed_pids)]


        #self.all_hits = self.all_hits[self.all_hits['pt']>2]
        return self.all_hits, self.hits # [908] #new_allowed_pids 

    def load_data_cms(self, file_number): 
        self._load_file_cms(file_number)   



        return self.all_hits, self.hits  # [908] #new_allowed_pids 
