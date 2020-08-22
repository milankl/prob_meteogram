import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np


def rotmat(rot):
    d = 2 * np.pi / 360.0  # assume input in degree
    return np.array([[np.cos(d * rot), -np.sin(d * rot)], [np.sin(d * rot), np.cos(d * rot)]])


def droplet(xy=(0.0, 0.0), width=0.4, height=1.0, rot=0.0):

    # convert to x,y being in the middle of the droplet
    x = xy[0] - width / 2.0
    y = xy[1] - height / 2.0

    verts = [
        (x + 0.5 * width, y + height),
        (x + 0.375 * width, y + 0.8 * height),
        (x, y + 0.5 * height),
        (x, y + 0.35 * height),
        (x, y),
        (x + width, y),
        (x + width, y + 0.35 * height),
        (x + width, y + 0.5 * height),
        (x + 0.625 * width, y + 0.8 * height),
        (x + 0.5 * width, y + height),
        (x + 0.5 * width, y + height),
    ]

    # perform rotation
    if rot != 0:
        R = rotmat(rot)
        # subtract xy to rotate around the droplets centre
        # and move it back to xy
        verts = R.dot(np.array(verts - np.array(xy)).T).T + np.array(xy)

    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    return Path(verts, codes)


if __name__ == "__main__":
    fig, ax = plt.subplots()
    patch = patches.PathPatch(droplet(), facecolor="aqua", lw=1)
    ax.add_patch(patch)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    plt.show()
    # plt.savefig("droplet.png", dpi=300)
    # plt.close(fig)
