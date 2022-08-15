import pdb
import tensorflow as tf 
import yaml 
import numpy as np 
import replay 
import globals 
import pandas as pd 


pd.set_option("precision", 16)

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

upper_bound = config['upper_bound']
lower_bound = config['lower_bound']


def policy(state, noise_object1, noise_object2):
    sampled_actions = tf.squeeze(globals.actor_model(state))
    noise1 = noise_object1()
    noise2 = noise_object2()
    # Adding noise to action
    #sampled_actions1 = sampled_actions.numpy() + noise1 
    sampled_actions1 = sampled_actions.numpy()[0] + noise1 
    sampled_actions2 = sampled_actions.numpy()[1] + noise2
    sampled_actions = np.array([sampled_actions1, sampled_actions2]).flatten()

    # We make sure action is within the boundaries of the environment 
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    print("legal action", legal_action)
    # now make sure it is one of the allowed hits
    new_action = find_closest_allowed_next_states(state, legal_action)
    # printing both to see that the action before correction is very off and does not learn 
    print("new action", new_action)
    return [np.squeeze(new_action)]




event = pd.read_csv(config['file_prefix']+ str(1000) + '.csv') 
# only one hit per layer and volume 
event = event.loc[event.groupby(['particle_id', 'volume_id', 'layer_id']).r.idxmin().values]
event = event.sort_values(['particle_id', 'r'])

tensor_particle_id = tf.convert_to_tensor(event.particle_id)
tensor_z = tf.convert_to_tensor(event.z)
tensor_r = tf.convert_to_tensor(event.r) 


def find_closest_allowed_next_states(state, suggested_action): 

    # find the intersect from the linear fit 
    close_hits = tf.convert_to_tensor(find_intersects(state[0][2:].numpy(), state[0][:2].numpy()))
    # find compatible hits that are closest to what the policy predicted (this is useless right now)
    distance_suggested = np.sqrt((close_hits[:,0] - suggested_action[0])**2 + (close_hits[:,1] - suggested_action[1])**2)
    index1 = tf.math.argmin(distance_suggested)
    safe_new_state = close_hits[index1]

    
    return tf.stack(safe_new_state)




# detector infor 
md = pd.read_csv(config['md_file'],  header=[0], index_col=[0, 1,2])
md = md.reset_index()
md.rename(columns = {'level_1':'volume_id', 'level_2':'layer_id'}, inplace = True)

# horizontal and vertical volume ids 
hor_vol = [8, 13, 17]
ver_vol = [7, 9, 12, 14, 16, 18]


def find_intersects(hit1, hit2): 
    """
    Extrapolates the straight line of two hits and finds intersections with the layers. 
    Then chooses the closest intersection to hit2 and finds the closest hits to this. 

    Inputs: Two hits, where hit2 is after hit1
    Outputs: A list of copmpatible hit positions
    """

    intersects = [] 

    #m = (hit2.r - hit1.r)/(hit2.z - hit1.z)
    #b = hit2.r - m*hit2.z

    m = (hit2[1] - hit1[1])/(hit2[0] - hit1[0])
    b = hit2[1] - m*hit2[0]


    # intersects of the straight line for the horizontal volumes 

    for vol in hor_vol: 
        sub = md[md['volume_id']==vol]

        for layer in sub.layer_id: 
            sub_layer = sub[sub['layer_id']==layer] 
            #print(sub_layer)
            z = (sub_layer['r_mean'].values -b)/m
            #print(z)
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

    intersects = intersects[intersects[:, 1].argsort()]
    intersects = intersects[(np.abs(intersects[:, 1]) > (np.abs(hit2[1])+0.1)) & 
                        (np.abs(intersects[:, 0]) > (np.abs(hit2[0])+0.1))]



    close_hits = [] 

    try: 
        interrow = intersects[0, ]
    except: 
        # this should generally not happen. There are a few tracks that go in towards the cetnre of the detector....
        interrow = [0, 0]
        print(intersects, hit1, hit2, m, b)


    # now find hits that are compatible with the intersection points 
    h_po = event[(event['volume_id']==interrow[2]) & (event['layer_id'] == interrow[3])]

    # the sensitivity will be changed 
    h_po = h_po[(np.abs(h_po['z']) <  np.abs(interrow[0]) + 0.5) & (np.abs(h_po['z']) > np.abs(interrow[0]) - 0.5) &
        (h_po['r'] < interrow[1] + 0.5) & (h_po['r'] > interrow[1] - 0.5)]

    distances = np.sqrt((h_po.z - interrow[0])**2 + (h_po.r - interrow[1])**2)
    if len(distances) > 0: 
        # five closest hits 
        index_five_closest =  np.argpartition(distances, 5)[:5]
        close_hits = h_po.iloc[index_five_closest, :][['z', 'r']].values
 
    else: 
        # if no compatible hits are found, just use the first intersection 
        close_hits = [[interrow[0], interrow[1]], [interrow[0], interrow[1]]]
        print("were here now and close hits are ", close_hits)
    
    #print(close_hits)

    return close_hits