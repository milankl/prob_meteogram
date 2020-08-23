"""
<<Alderaan>> example
but without the tweaks in original `meteogram_alderaan.py`.
"""
import datetime

import matplotlib.pyplot as plt
import numpy as np

from prob_meteogram import Loc, prob_meteogram


# %% Prepare

# set location
loc = Loc(
    latitude=-33.4,
    longitude=-70.7,
    address="Alderaan City, Alderaan"
)

# READ DATA (provided in repo)
time_raw = np.load("data/time.npz")["arr_0"]

# time is the first dim in these
t = np.load("data/temperature.npz")["arr_0"]
lcc = np.load("data/low_clouds.npz")["arr_0"]
mcc = np.load("data/medium_clouds.npz")["arr_0"]
hcc = np.load("data/high_clouds.npz")["arr_0"]
lsp = np.load("data/precip.npz")["arr_0"]
u = np.load("data/uwind.npz")["arr_0"]
v = np.load("data/vwind.npz")["arr_0"]

# convert time to datetime objects
datetime0 = datetime.datetime(1900, 1, 1)
time_utc = [datetime0 + datetime.timedelta(hours=int(t)) for t in time_raw]

# group data
data = {
    "time_utc": time_utc,
    "t": t,
    "lcc": lcc,
    "mcc": mcc,
    "hcc": hcc,
    "lsp": lsp,
    "u": u,
    "v": v,
}


# %% Call the function

prob_meteogram(loc, data)


plt.show()
