"""
Shared utilities for the patch drawings.
"""

import numpy as np

def rotmat(rot):
    d = 2 * np.pi / 360.0  # assume input in degree
    return np.array([[np.cos(d * rot), -np.sin(d * rot)], [np.sin(d * rot), np.cos(d * rot)]])
