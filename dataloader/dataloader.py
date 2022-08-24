import numpy as np
import pandas as pd 



unique_layer_id_mapping = pd.read_csv('/home/lhv14/unique_layer_id_mapping.csv') 
allowed_layer_connections = pd.read_csv('/home/lhv14/allowed_layer_connections.csv')

prefix = '/home/lhv14/exatrkx/Tracking-ML-Exa.TrkX/alldata/train_1_processed1000/processed_event00000' 

class DataLoader:
    """Preprocesses and loads data. Returns processed hits"""  

    def __init__(self): 
        self.hits = pd.DataFrame({})

    def _load_file(self, filenumber): 
        self.hits = pd.read_csv(prefix+str(1000)+'.csv') 

    def _pt_cut(self, pt_min): 
        self.hits = self.hits[self.hits['pt'] < pt_min] 

    def _filter_by_num_track_hits(self, min_hits): 
        c = self.hits.groupby(['particle_id']).count()
        self.hits = self.hits[self.hits['particle_id'].isin(c[c['hit_id'] > min_hits].index)]

    def _assign_unique_layer_id(self): 
        self.hits = self.hits.merge(unique_layer_id_mapping, on=['volume_id', 'layer_id'], how='left')
        self.hits = self.hits.sort_values(['r', 'z'])
    
    def _get_layer_module_connections(self): 
        self.hits[['next_mod_id', 'connecting_layer_id']] = self.hits.groupby('particle_id')['discrete_module_id', 'unique_layer_id'].shift(-1)

    def _filter_out_missing_hits(self): 
        pass 

    def _filterout_na(self): 
        self.hits = self.hits.dropna() 

    def load_data(self, file_number): 
        self._load_file(file_number)
        self._filter_by_num_track_hits(6) 
        self._assign_unique_layer_id() 
        self._get_layer_module_connections()
        self._filterout_na() 
        
        return self.hits 
