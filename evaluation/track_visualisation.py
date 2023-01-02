from tkinter import N
from turtle import window_width
import pyglet 
import pandas as pd 
import numpy as np 
from pyglet import shapes
from pyglet.gl import glClearColor
from pyglet import clock

rf_file = pd.read_csv('garage_outputs.csv')
print(rf_file)

batch = pyglet.graphics.Batch()
batch2 = pyglet.graphics.Batch() 

window_length = 1000
window_height = 1000 

scale_z = window_length/(2*300) 
scale_r = window_height/120

rf_file['mc_z'] = (rf_file['mc_z'].values+300) * scale_z 
rf_file['mc_r'] = rf_file['mc_r'].values*scale_r 
rf_file['pred_z'] =(rf_file['pred_z'].values+300)*scale_z
rf_file['pred_r'] = rf_file['pred_r'].values*scale_r


#p1 = rf_file[rf_file['particle_id']==-18951]

n_track = 1
# sample every 100th particle id 
# keep it sorted so it's easier to compare to csv 
pid_indices = np.unique(rf_file.particle_id, return_index=True)[1]

pids = rf_file.particle_id.values[pid_indices][-3000:][::n_track]


#pids = rf_file.particle_id.values[::60]
files = rf_file.filenumber.values[pid_indices][::n_track]
#print(pids, files, len(pids), len(files))
window = pyglet.window.Window(window_length, window_height)

#detector = pd.read_csv('/home/lhv14/exatrkx/Tracking-ML-Exa.TrkX/data/detectors.csv')
detector = pd.read_csv('/home/lhv14/detector_fixed_endcaps.csv')

#detector['cr'] = np.sqrt(detector.cx**2 + detector.cy**2)/10
#detector['cz'] = detector['cz']/10



detector['cz'] = (detector['cz'].values+300)*scale_z 
detector['cr'] = detector['cr'].values*scale_r 


md = (
    detector.groupby(["layer_id", "volume_id"])["cz", "cr"]
    .agg(["min", "max"])
    .reset_index()
)


lines = [] 
for i in range(md.shape[0]): 
    mdrow = md.iloc[i,]
    #print(md1)
    line = shapes.Line(mdrow['cz']['min'], mdrow['cr']['min'], mdrow['cz']['max'], mdrow['cr']['max'], 5, color = (200, 200, 200), batch = batch)
    line.opacity = 100
    lines.append(line)




class Point:
    def __init__(self): 
        self.circles = [] 
        self.text = [0]
        self.i = 0  
        self.pid_counter = 0 
        self.pid = pids[0]
        self.filenumber = files[0]
        self.label = [] 
        self.particle = rf_file[(rf_file['particle_id']==self.pid)]
        n_track_hits = len(self.particle)
    
    def plot_point(self, dt): 
        #self.text = []
        self.particle = rf_file[(rf_file['particle_id']==self.pid)]
        n_track_hits = len(self.particle)

        #print(str(self.particle.reward.values[2:]))    
        #self.plot_label()
        
        color1 = 160
        #print("i is now ", self.i)
        if self.i < (n_track_hits-1):
         #print(self.particle)
         hit = self.particle.iloc[self.i, ]   
         self.circles.append(shapes.Circle(hit.mc_z, hit.mc_r, 5, color=(color1,60,60), batch=batch)) 
         #self.circles.append(pyglet.text.Label("Particle id:  " + str(self.pid) + "  After training on " + str(self.pid_counter*10) +"tracks", font_size=12, batch=batch))
         #self.circles.append(pyglet.text.Label("reward:  " + str(self.particle.reward.values[2:]), font_size=12, batch=batch))
         #self.circles.append(shapes.Circle(self.i*10, 200, 50, color=(color1,60,60), batch=batch)) 

         self.i += 1 
         #text.delete()


        elif (self.i > (n_track_hits-2)) & (self.i < (n_track_hits*2-2)): 
            hit = self.particle.iloc[self.i-(n_track_hits-1), ]
            color3 = 2014
            self.circles.append(shapes.Circle(hit.pred_z, hit.pred_r, 5, color=(0,60,color3), batch=batch)) 
            self.i+=1 


        else: 
            self.i = 0 
            self.pid_counter += 1 
            self.pid = pids[self.pid_counter]
            self.filenumber = files[self.pid_counter]
            #del(self.text)
            self.circles = []
            #self.reset_label() 
    

            #self.particle = rf_file[rf_file['particle_id']==self.pid]
    def get_label(self): 
        #print(self.particle)
        #print(self.particle.reward.values[2:])
        return pyglet.text.Label("reward:  " + str(self.particle.reward.values[2:]), font_size=12)

        



p = Point() 

clock.schedule_interval(p.plot_point, 0.2)
frame = 0 
@window.event
def on_draw():
    global frame 
    window.clear()
    #batch2.draw() 
    frame += 1 
    batch.draw() 
    label = p.get_label()
    label.draw()
    # par = rf_file[(rf_file['particle_id']==pids[frame])]

    # text = pyglet.text.Label("reward:  "+ str(par.reward.values[2:]), font_size=12, batch=batch2)
    batch2.draw()
    # text.delete()

    pyglet.image.get_buffer_manager().get_color_buffer().save('../screenshots/screenshot'+str(frame)+'.png')    #label.draw()
    #image_count += 1 


pyglet.app.run()
