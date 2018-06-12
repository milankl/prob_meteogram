import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

def rotmat(rot):
    d = 2*np.pi/360.    # assume input in degree
    return np.array([[np.cos(d*rot),-np.sin(d*rot)],[np.sin(d*rot),np.cos(d*rot)]])
        
def droplet(xy=(0.,0.),width=0.4,height=1.,rot=0.):
    
    
    # convert to x,y being in the middle of the droplet
    x = xy[0]-width/2.
    y = xy[1]-height/2.
    
    verts = [
    (x+.5*width, y+height),
    (x+.375*width,y+.8*height),
    (x, y+.5*height),
    (x, y+.35*height),
    (x, y),
    (x+width, y),
    (x+width,y+.35*height),
    (x+width,y+.5*height),
    (x+.625*width,y+.8*height),
    (x+.5*width,y+height),
    (x+.5*width,y+height)]
    
    # perform rotation
    if rot != 0:
        R = rotmat(rot)
        # subtract xy to rotate around the droplets centre
        # and move it back to xy
        verts = R.dot(np.array(verts-np.array(xy)).T).T + np.array(xy)
    
    codes = [Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CLOSEPOLY]
    
    return Path(verts,codes)