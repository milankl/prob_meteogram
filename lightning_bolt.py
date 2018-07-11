import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np
        
def lightning_bolt(xy=(0.,0.),width=0.6,height=1.,rot=0.):
        
    # convert to x,y being in the middle of the droplet
    x = xy[0]-width/2
    y = xy[1]-height/2
    
    h1 = 0.7*height
    h2 = 0.55*height
    
    verts = [
    (x,y),
    (x+width/2,y+h2),
    (x,y+h2),
    (x+width/2,y+height),
    (x+width,y+height),
    (x+width/2,y+h1),
    (x+width,y+h1),
    (x,y)
    ]
    
    codes = [Path.MOVETO]+[Path.LINETO]*(len(verts)-2)+[Path.CLOSEPOLY]
    
    return Path(verts,codes)
    
    
# fig,ax = plt.subplots(1,1)
# patch = patches.PathPatch(lightning_bolt(), facecolor='yellow', lw=1)
# ax.add_patch(patch)
# 
# ax.set_xlim(-1,1)
# ax.set_ylim(-1,1)
# plt.savefig("lightning_bolt.png",dpi=300)
# plt.close(fig)