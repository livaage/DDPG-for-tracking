import numpy as np
import pandas as pd 




def find_df_hits(hits, hit1, hit2): 
    hit1_df = hits[(hits['z'] == hit1[0]) & (hits['r'] == hit1[1])].squeeze()
    hit2_df = hits[(hits['z'] == hit2[0]) & (hits['r'] == hit2[1])].squeeze()
    return hit1_df, hit2_df
