from collections import namedtuple
import datetime
import os, sys, inspect
import warnings

HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(HERE_PATH)

from geopy.geocoders import Nominatim
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.dates import (
    WeekdayLocator,
    DayLocator,
    HourLocator,
    DateFormatter,
    drange,
    date2num,
    num2date,
)
from matplotlib.patches import Circle, Ellipse, Rectangle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from netCDF4 import Dataset
import numpy as np
import pytz
from scipy.interpolate import interp1d
from tzwhere import tzwhere  # * somewhat heavy dependency

# from .sunrise import Sun
from sunrise import Sun

# from .patches import droplet, wind_sock, lightning_bolt
from patches import droplet, wind_sock, lightning_bolt


Loc = namedtuple("Loc", "latitude longitude address")
LOC_DEFAULT = Loc(-22.9, -43.2, "Rio de Janeiro, Brazil")

DEGREE_SIGN = "\N{DEGREE SIGN}"  # also can use `°` directly

TZ = tzwhere.tzwhere()


def find_closest_ind(x, a):
    """Finds the closest index in `x` to a given value `a`."""
    return np.argmin(abs(x - a))


def convert_longitude(lon):
    """Convert `lon` from [-180, 180] form to [0, 360] form."""
    if lon < 0:
        return 360.0 + lon
    else:
        return lon


def lon_string(lon):
    """String version of `lon` value.

    e.g., '20.0°E'
    """
    if lon >= 0:
        return f"{lon:.1f}{DEGREE_SIGN}E"
    else:
        return f"{abs(lon):.1f}{DEGREE_SIGN}W"


def lat_string(lat):
    """String version of `lat` value.

    e.g., '40.0°N'
    """
    if lat > 0:
        return f"{lat:.1f}{DEGREE_SIGN}N"
    else:
        return f"{abs(lat):.1f}{DEGREE_SIGN}S"


def timezone_offset(loc, date):
    """Calculate UTC offset for given location and date."""
    timezone_str = TZ.tzNameAt(loc.latitude, loc.longitude)
    timezone = pytz.timezone(timezone_str)
    utcoffset = timezone.utcoffset(date)
    return utcoffset


def sunrise_sunset(loc, date, utcoffset):
    """Calculate sunrise/set for given location, date, and UTC offset."""
    sun = Sun(lat=loc.latitude, long=loc.longitude)
    t_sunrise = sun.sunrise(when=date)
    t_sunset = sun.sunset(when=date)

    # convert from time to datetime object
    t_sunrise = datetime.datetime(2000, 1, 1, t_sunrise.hour, t_sunrise.minute)
    t_sunset = datetime.datetime(2000, 1, 1, t_sunset.hour, t_sunset.minute)

    # add utcoffset
    t_sunrise = (t_sunrise + utcoffset).time()
    t_sunset = (t_sunset + utcoffset).time()

    return t_sunrise, t_sunset


def sunrise_string(loc, date, utcoffset):
    """Create sunrise/set string with times and Sun symbol."""
    sunsymb = "\u263C"
    arrowup = "\u2191"
    arrowdn = "\u2193"

    sunrise, sunset = sunrise_sunset(loc, date, utcoffset)
    sunrise_str = "{:0=2d}:{:0=2d}".format(sunrise.hour, sunrise.minute)
    sunset_str = "{:0=2d}:{:0=2d}".format(sunset.hour, sunset.minute)

    return sunsymb + arrowup + sunrise_str + arrowdn + sunset_str


#  axes formatting
def cloud_ax_format(ax, dates, loc):
    """Set up the cloud ax."""
    loc_name = loc.address.split(",")[0]
    ax.set_title(
        f"Meteogram {loc_name} ({lat_string(loc.latitude)}, {lon_string(loc.longitude)})",
        loc="left",
        fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([dates[0], dates[-1]])
    ax.set_ylim([-0.2, 1])


def rain_ax_format(ax, dates, rain_explanation, snow, dsize=78, dstring=(2, 0, 45)):
    """Set up the rain ax."""
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    # ax.xaxis.grid(alpha=0.2)
    # ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(7)))
    # ax.set_xticklabels([])
    ax.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 6)))  # minor
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(5)))
    ax.get_xaxis().set_tick_params(which="minor", direction="in")
    ax.get_xaxis().set_tick_params(which="major", direction="in")
    ax.set_xticklabels([])

    bg_rect = Rectangle(
        (0.84, 0),
        0.16,
        1.0,
        linewidth=0.3,
        edgecolor="k",
        facecolor="w",
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(bg_rect)

    # distinguish between rain and snow
    if np.all(snow):  # only snowfall
        ax.scatter(
            [0.917, 0.917],
            [0.25, 0.5],
            dsize * 0.6,
            color=rain_explanation,
            transform=ax.transAxes,
            marker=(6, 2, 0),
            zorder=3,
        )
        ax.text(
            0.992,
            0.70,
            "snowfall",
            fontsize=8,
            fontweight="bold",
            transform=ax.transAxes,
            ha="right",
        )
        ax.text(0.992, 0.45, "very likely", fontsize=8, transform=ax.transAxes, ha="right")
        ax.text(0.992, 0.2, "less likely", fontsize=8, transform=ax.transAxes, ha="right")
    elif np.all(~snow):  # only rainfall
        ax.scatter(
            [0.917, 0.917],
            [0.25, 0.5],
            dsize,
            color=rain_explanation,
            transform=ax.transAxes,
            marker=droplet(rot=-30),
            zorder=3,
        )
        ax.text(
            0.992,
            0.70,
            "rainfall",
            fontsize=8,
            fontweight="bold",
            transform=ax.transAxes,
            ha="right",
        )
        ax.text(0.992, 0.45, "very likely", fontsize=8, transform=ax.transAxes, ha="right")
        ax.text(0.992, 0.2, " less likely", fontsize=8, transform=ax.transAxes, ha="right")
    else:  # mix of snow and rain
        ax.scatter(
            [0.919, 0.919],
            [0.16, 0.41],
            dsize,
            color=rain_explanation,
            transform=ax.transAxes,
            marker=droplet(rot=-30),
            zorder=3,
        )
        ax.scatter(
            [0.914, 0.914],
            [0.24, 0.49],
            dsize * 0.42,
            color=rain_explanation,
            transform=ax.transAxes,
            marker=(6, 2, 0),
            zorder=3,
        )
        ax.text(
            0.992,
            0.61,
            "snow or\n rainfall",
            fontsize=8,
            fontweight="bold",
            transform=ax.transAxes,
            ha="right",
        )
        ax.text(0.992, 0.4, "very likely", fontsize=8, transform=ax.transAxes, ha="right")
        ax.text(0.992, 0.15, " less likely", fontsize=8, transform=ax.transAxes, ha="right")

    ax.text(0.84, 0.75, "\u25B8 heavy", fontsize=8, transform=ax.transAxes, ha="left")
    ax.text(0.84, 0.43, "\u25B8 medium", fontsize=8, transform=ax.transAxes, ha="left")
    ax.text(0.84, 0.11, "\u25B8 light", fontsize=8, transform=ax.transAxes, ha="left")


def temp_ax_format(ax, tminmax, dates, utcoffset, loc):
    """Set up the temperature ax."""
    ax.text(
        0.01, 0.92, sunrise_string(loc, dates[0], utcoffset), fontsize=10, transform=ax.transAxes
    )

    td = tminmax[1] - tminmax[0]
    if td >= 13.0:
        ddegree = 3
    elif td < 13.0 and td >= 5.0:
        ddegree = 2
    else:
        ddegree = 1

    ax.set_yticks(np.arange(np.round(tminmax[0]) - 3, np.round(tminmax[1]) + 3, ddegree))
    ax.set_ylim(np.round(tminmax[0]) - 3, np.round(tminmax[1]) + 3)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%d" + "\N{DEGREE SIGN}" + "C"))

    # x axis lims, ticks, labels
    ax.set_xlim(dates[0], dates[-1])
    ax.xaxis.set_minor_locator(HourLocator(np.arange(6, 25, 6)))  # minor
    ax.xaxis.set_minor_formatter(DateFormatter("%Hh"))
    ax.get_xaxis().set_tick_params(which="minor", direction="out", pad=2, labelsize=6)
    ax.grid(alpha=0.2)

    # major weekdays not weekend in black
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(5)))
    ax.xaxis.set_major_formatter(DateFormatter(" %a\n %d %b"))
    plt.setp(ax.get_xticklabels(), ha="left")
    ax.get_xaxis().set_tick_params(which="major", direction="out", pad=10, labelsize=10)

    # major weekends in blue
    ax_weekend = ax.twiny()
    ax_weekend.set_xlim(dates[0], dates[-1])
    ax_weekend.xaxis.set_major_locator(WeekdayLocator(byweekday=[5, 6]))
    ax_weekend.xaxis.set_major_formatter(DateFormatter(" %a\n %d %b"))
    ax_weekend.xaxis.tick_bottom()
    plt.setp(ax_weekend.get_xticklabels(), ha="left", color="C0")
    ax_weekend.get_xaxis().set_tick_params(which="major", direction="out", pad=10, labelsize=10)
    ax_weekend.grid(alpha=0.2)

    # remove labels at edges
    if dates[-1].hour < 13:  # remove only if there is not enough space
        if dates[-1].weekday() > 4:  # weekend
            ax_weekend.get_xticklabels()[-1].set_visible(False)
        else:  # during the week
            ax.get_xticklabels()[-1].set_visible(False)

    # ax.get_xticklabels()[2].set_color("C0") #TODO make automatic
    # ax.get_xticklabels()[3].set_color("C0")
    # ax.get_xticklabels(which="minor")[-1].set_visible(False)
    # ax.get_xticklabels(which="minor")[0].set_visible(False)


def wind_ax_format(ax, dates):
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(0.2, 2.5)
    ax.set_yticks([])
    # ax.xaxis.set_minor_locator(HourLocator(np.arange(0, 25, 6)))    # minor
    # ax.xaxis.set_major_locator(WeekdayLocator(byweekday=range(5)))
    # ax.get_xaxis().set_tick_params(which='minor', direction='in')
    # ax.get_xaxis().set_tick_params(which='major', direction='in')
    ax.set_xticklabels([])


def create_fig():
    fig = plt.figure(figsize=(10, 4))

    # subplots adjust
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 6], hspace=0, figure=fig)
    cloud_ax = fig.add_subplot(gs[0])
    rain_ax = fig.add_subplot(gs[1])
    temp_ax = fig.add_subplot(gs[2])

    fig.tight_layout(rect=[0.02, 0.03, 1, 0.97])
    # use constrained_layout instead?


if __name__ == "__main__":
    # p = droplet()
    # print(p)

    create_fig()
    plt.show()
