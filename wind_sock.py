import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np

def rotmat(rot):
    d = 2*np.pi/360.    # assume input in degree
    return np.array([[np.cos(d*rot),-np.sin(d*rot)],[np.sin(d*rot),np.cos(d*rot)]])
        
def wind_sock(xy=(0.,0.),width=0.4,height=.8,rot=35):
    
    
    # convert to x,y being in the middle of the droplet
    x = xy[0]-width/2
    y = xy[1]-height/2
    
    polewidth = width*0.03
    
    verts_pole = [
    (x, y),
    (x, y+height),
    (x+polewidth, y+height),
    (x+polewidth,y),
    (x,y)
    ]
    
    sock_heigth = 0.8
    sock_width1 = 0.2
    sock_width6 = 0.15
    sock_dist1 = sock_heigth*5/6
    sock_dist2 = sock_heigth*4/6
    sock_dist3 = sock_heigth*3/6
    sock_dist4 = sock_heigth*2/6
    sock_dist5 = sock_heigth/6

    x = 0.
    y = -sock_heigth
    
    sw = lambda sd: sock_width6 + 2*sd/sock_dist1*(sock_width1 - sock_width6)/2.
    
    sock_width2 = sw(sock_dist2)
    sock_width3 = sw(sock_dist3)
    sock_width4 = sw(sock_dist4)
    sock_width5 = sw(sock_dist5)
    
    verts_sock1 = [
    (x+sock_width1/2,y+sock_dist1),
    (x,y+sock_heigth),
    (x-sock_width1/2,y+sock_dist1),
    (x+sock_width1/2,y+sock_dist1),
    (x+sock_width2/2,y+sock_dist2),
    (x-sock_width2/2,y+sock_dist2),
    (x-sock_width3/2,y+sock_dist3),
    (x-sock_width1/2,y+sock_dist1),
    (x,y+sock_heigth),
    (x+sock_width1/2,y+sock_dist1)
    ]
    
    verts_sock2 = [
    (x+sock_width2/2,y+sock_dist2),
    (x+sock_width3/2,y+sock_dist3),
    (x-sock_width3/2,y+sock_dist3),
    (x-sock_width4/2,y+sock_dist4),
    (x+sock_width4/2,y+sock_dist4),
    (x+sock_width5/2,y+sock_dist5),
    (x+sock_width2/2,y+sock_dist2)
    ]
    
    verts_sock3 = [
    (x-sock_width4/2,y+sock_dist4),
    (x-sock_width6/2,y),
    (x+sock_width6/2,y),
    (x+sock_width5/2,y+sock_dist5),
    (x-sock_width5/2,y+sock_dist5),
    (x-sock_width4/2,y+sock_dist4)
    ]
    
    codes_pole = [Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY]
    
    codes_sock1 = [Path.MOVETO]+[Path.LINETO]*(len(verts_sock1)-2)+[Path.CLOSEPOLY]
    codes_sock2 = [Path.MOVETO]+[Path.LINETO]*(len(verts_sock2)-2)+[Path.CLOSEPOLY]
    codes_sock3 = [Path.MOVETO]+[Path.LINETO]*(len(verts_sock3)-2)+[Path.CLOSEPOLY]
    
    xys = [xy[0]-width/2+polewidth,xy[1]+height/2]
    
    if rot != 0:
        R = rotmat(rot)
        verts_sock1 = R.dot(np.array(verts_sock1).T).T + np.array(xys)
        verts_sock2 = R.dot(np.array(verts_sock2).T).T + np.array(xys)
        verts_sock3 = R.dot(np.array(verts_sock3).T).T + np.array(xys)
    
    
    verts = verts_pole+[(a,b) for a,b in verts_sock1]+[(a,b) for a,b in verts_sock2]+[(a,b) for a,b in verts_sock3]
    codes = codes_pole+codes_sock1+codes_sock2+codes_sock3
    
    return Path(verts,codes)
    

# fig,ax = plt.subplots(1,1)
# patch = patches.PathPatch(wind_sock(), facecolor='C3', lw=1)
# ax.add_patch(patch)
# 
# ax.set_xlim(-1,1)
# ax.set_ylim(-1,1)
# plt.savefig("wind_sock.png",dpi=300)
# plt.close(fig)