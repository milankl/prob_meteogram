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

from .sunrise import Sun
from .patches import droplet, wind_sock, lightning_bolt


Loc = namedtuple("Loc", "latitude longitude address")
LOC_DEFAULT = Loc(-22.9, -43.2, "Rio de Janeiro, Brazil")

DEGREE_SIGN = "\N{DEGREE SIGN}"  # also can use `°` directly

TZ = tzwhere.tzwhere()
# TODO: this ^ takes a bit so should only do it when we need it

SPLINE_RES = 360  # number of values (evenly spaced) for the interpolated data


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


def spline_dates(dates, resolution=SPLINE_RES):
    """Create target date coordinate for the interpolated data."""
    numdates = date2num(dates)
    numdates_spline = np.linspace(numdates[0], numdates[-1], num=resolution)
    return num2date(numdates_spline)


def spline_data_by_date(data, numdates, resolution=SPLINE_RES):
    """
    Interpolate the data.

    numdates : 
        original time coordinate, in number form
    """
    numdates_spline = np.linspace(numdates[0], numdates[-1], num=resolution)
    data_spline = np.empty((resolution, data.shape[1]))
    for e in range(data.shape[1]):
        spline = interp1d(numdates, data[:, e], kind="cubic")
        data_spline[:, e] = spline(numdates_spline)
    return data_spline


def rain_prob(lsp, rdates, *, color=None):
    """
    Calculate precipitation probability.

    """
    if color is None:
        color = [0.0, 0.17, 0.41]

    # in mm
    rthresh = [0.05, 0.5, 4.0, 10.0]  # light, medium, heavy+very heavy
    bins = np.array(
        [min(0, lsp.min()), rthresh[0], rthresh[1], rthresh[2], max(rthresh[3], lsp.max())]
    )

    # preallocate probablity per rainfall category
    P = np.empty((len(bins) - 1, len(rdates)))
    for i in range(len(rdates)):
        P[:, i], _ = np.histogram(lsp[i, :], bins)

    P = P / lsp.shape[1]  # normalize by number of ensemble members

    # turn into alpha values
    colormat = np.zeros((P.shape[1], 4))
    colormat[:, 0] = color[0]  # RGB values of "C0" matplotlib standard
    colormat[:, 1] = color[1]
    colormat[:, 2] = color[2]

    lightrain = colormat.copy()
    medrain = colormat.copy()
    heavyrain = colormat.copy()

    lightrain[:, 3] = P[1, :]
    medrain[:, 3] = P[2, :]
    heavyrain[:, 3] = P[3, :]

    rain_explanation = colormat[:2, :]
    rain_explanation[:, -1] = [0.2, 1.0]  # values for example transparency

    return lightrain, medrain, heavyrain, rain_explanation


def storm(spd):
    """
    Compute some things about storminess.

    spd : 
        wind speed (m/s)
    """
    threshold1 = 8.0  # m/s   5 Beaufort
    threshold2 = 10.7  # 6 Beaufort
    threshold3 = 13.8  # 7 Beaufort

    p_storm = np.mean(spd >= threshold1, axis=1)

    s_storm = np.zeros_like(p_storm)
    for i, s in enumerate(spd):
        st = s >= threshold1
        if np.sum(st):
            s_storm[i] = np.median(s[st])

    storm_strength = np.zeros((3, s_storm.shape[0]))
    storm_strength[0, :] = 1.0 * (np.logical_and(s_storm >= threshold1, s_storm < threshold2))
    storm_strength[1, :] = 1.0 * (np.logical_and(s_storm >= threshold2, s_storm < threshold3))
    storm_strength[2, :] = 1.0 * (s_storm >= threshold3)

    color = [0.71, 0.12, 0.12]

    # turn into alpha values
    colormat = np.zeros((p_storm.shape[0], 4))
    colormat[:, 0] = color[0]
    colormat[:, 1] = color[1]
    colormat[:, 2] = color[2]

    colormat[:, 3] = p_storm

    return colormat, storm_strength.astype(np.bool)


def lightning(lcc, mcc, hcc, tmedian):
    # fake some lightning chance
    tcc = (0.5 * mcc + 2 * lcc) / 3.0
    cthresh = 0.3
    p_light = np.mean(tcc > cthresh, axis=1)

    p_light[p_light >= 0.5] = 1.0
    p_light[np.logical_and(p_light >= 0.1, p_light < 0.5)] = 0.7
    p_light[p_light < 0.1] = 0.0

    # only for warm temperaturs
    p_light[tmedian < 16] = 0.0

    # turn into alpha values
    colormat = np.zeros((p_light.shape[0], 4))
    colormat[:, 0] = 1.0
    colormat[:, 1] = 1.0
    colormat[:, 2] = 0.0

    colormat[:, 3] = p_light

    return colormat


def snow_or_rain(tmedian):
    # fake snow information based on temperature
    snow = tmedian[:-1] < 0  # precipitation has one time step less
    return snow


def temp_plotter(ax, dates, t_data_spline, tminmax, color="white", alpha=0.08):

    # these temperatures will be associated with the lower and upper end of the colormap
    clev = [-10, 20]
    tmin, tmax = (-100, 100)  # absurdly low and high temperatures
    cmap = "rainbow"
    cmat = [[clev[0], clev[0]], [clev[1], clev[1]]]
    cmat_high = clev[1] * np.ones((2, 2))
    cmat_low = clev[0] * np.ones((2, 2))

    ax.contourf([dates[0], dates[-1]], clev, cmat, 128, cmap=cmap)
    ax.contourf(
        [dates[0], dates[-1]],
        [clev[1], tmax],
        cmat_high,
        128,
        vmin=clev[0],
        vmax=clev[1],
        cmap=cmap,
    )
    ax.contourf(
        [dates[0], dates[-1]], [tmin, clev[0]], cmat_low, 128, vmin=clev[0], vmax=clev[1], cmap=cmap
    )

    numtime = date2num(spline_dates(dates))

    # TODO compare to 1std, 2std whether luminace data is representative

    n_tsteps = len(numtime)
    n_ens_members = t_data_spline.shape[1]
    ylim = ax.get_ylim()

    # plot first half of ensemble members from the bottom, the first without transparency
    ax.fill_between(
        numtime, ylim[0] * np.ones(n_tsteps), t_data_spline[:, 0], facecolor=color, alpha=1.0
    )
    for i in range(1, n_ens_members // 2):
        ax.fill_between(
            numtime, ylim[0] * np.ones(n_tsteps), t_data_spline[:, i], facecolor=color, alpha=alpha
        )

    # and the second half from the top, the last without transparency
    for i in range(n_ens_members // 2, n_ens_members):
        ax.fill_between(
            numtime, t_data_spline[:, i], ylim[1] * np.ones(n_tsteps), facecolor=color, alpha=alpha
        )

    ax.fill_between(
        numtime, t_data_spline[:, -1], ylim[1] * np.ones(n_tsteps), facecolor=color, alpha=1.0
    )


def clouds_plotter(axis, dates, highcloud, midcloud, lowcloud, interp=True):
    """ Adds the different types of clouds to a given axis."""
    # add sun (and moon?)
    for t in np.arange(len(dates)):
        idate = datetime.datetime(2018, 6, 7, 12, 0)
        while idate < dates[-1]:
            # sun = Circle((dates[t], 0.5), 0.2, color='yellow', zorder=0)
            sun = Ellipse((idate, 0.5), 0.4 / 2.0, 0.5, angle=0.0, color="yellow", zorder=0)
            axis.add_artist(sun)
            idate = idate + datetime.timedelta(1)

    # interpolate dates (if set)
    if interp:
        dates = spline_dates(dates)

    # add mean cloud covers and scale to [0...1]
    highcloudm = np.median(highcloud, axis=1)
    midcloudm = np.median(midcloud, axis=1)
    lowcloudm = np.median(lowcloud, axis=1)

    totalcloud = (highcloudm + midcloudm + lowcloudm) / 3.0
    totalcloudhalf = totalcloud / 2.0
    lowerbound = -totalcloudhalf + 0.5
    upperbound = totalcloudhalf + 0.5

    # don't plot clouds where totalcloud <= e.g. 0.05
    threshold = 0.05

    # highcloud light grey, lowcloud dark grey
    axis.fill_between(
        dates,
        y1=lowerbound,
        y2=upperbound,
        color="0.95",
        zorder=1,
        alpha=0.8,
        edgecolor="none",
        where=totalcloud >= threshold,
    )
    axis.fill_between(
        dates,
        y1=lowerbound,
        y2=upperbound - highcloudm / 3.0,
        color="0.7",
        zorder=2,
        alpha=0.6,
        edgecolor="none",
        where=totalcloud >= threshold,
    )
    axis.fill_between(
        dates,
        y1=lowerbound,
        y2=lowerbound + lowcloudm / 3.0,
        color="0.4",
        zorder=3,
        alpha=0.3,
        edgecolor="none",
        where=totalcloud >= threshold,
    )
    axis.set_facecolor("lightskyblue")


def rain_plotter(
    ax, lightrain, medrain, heavyrain, snow, rdates, dsize=78, ssize=55, dstring=(2, 0, 45)
):

    dt = datetime.timedelta(hours=0.9)  # used to shift symbols left/right
    dropletpath = droplet(rot=-30)
    snowmarker = (6, 2, 0)

    # light snow
    ax.scatter(
        [d for d, s in zip(rdates, snow) if s],
        np.zeros_like(rdates)[snow],
        ssize * 0.8,
        color=lightrain[snow, :],
        marker=snowmarker,
    )

    # light rain
    ax.scatter(
        [d for d, s in zip(rdates, snow) if ~s],
        np.zeros_like(rdates)[~snow],
        dsize * 0.9,
        color=lightrain[~snow, :],
        marker=dropletpath,
    )

    # medium snow
    ax.scatter(
        [d + 1.3 * dt for d, s in zip(rdates, snow) if s],
        1.06 + np.zeros_like(rdates)[snow],
        ssize,
        color=medrain[snow, :],
        marker=snowmarker,
    )
    ax.scatter(
        [d - 1.3 * dt for d, s in zip(rdates, snow) if s],
        0.94 + np.zeros_like(rdates)[snow],
        ssize,
        color=medrain[snow, :],
        marker=snowmarker,
    )

    # medium rain
    ax.scatter(
        [d + dt for d, s in zip(rdates, snow) if ~s],
        1.05 + np.zeros_like(rdates)[~snow],
        dsize,
        color=medrain[~snow, :],
        marker=dropletpath,
    )
    ax.scatter(
        [d - dt for d, s in zip(rdates, snow) if ~s],
        0.95 + np.zeros_like(rdates)[~snow],
        dsize,
        color=medrain[~snow, :],
        marker=dropletpath,
    )

    # heavy snow
    ax.scatter(
        [d - 0.6 * dt for d, s in zip(rdates, snow) if s],
        2.18 + np.zeros_like(rdates)[snow],
        ssize,
        color=heavyrain[snow, :],
        marker=snowmarker,
    )
    ax.scatter(
        [d + 1.3 * dt for d, s in zip(rdates, snow) if s],
        1.88 + np.zeros_like(rdates)[snow],
        ssize * 0.9,
        color=heavyrain[snow, :],
        marker=snowmarker,
    )
    ax.scatter(
        [d - 1.3 * dt for d, s in zip(rdates, snow) if s],
        1.78 + np.zeros_like(rdates)[snow],
        ssize * 0.8,
        color=heavyrain[snow, :],
        marker=snowmarker,
    )

    # heavy rain
    ax.scatter(
        [d for d, s in zip(rdates, snow) if ~s],
        2.1 + np.zeros_like(rdates)[~snow],
        dsize * 1.1,
        color=heavyrain[~snow, :],
        marker=dropletpath,
    )
    ax.scatter(
        [d + dt for d, s in zip(rdates, snow) if ~s],
        1.87 + np.zeros_like(rdates)[~snow],
        dsize * 1.1,
        color=heavyrain[~snow, :],
        marker=dropletpath,
    )
    ax.scatter(
        [d - 2.2 * dt for d, s in zip(rdates, snow) if ~s],
        1.95 + np.zeros_like(rdates)[~snow],
        dsize * 1.1,
        color=heavyrain[~snow, :],
        marker=dropletpath,
    )


def wind_plotter(ax, dates, p_storm, storm_strength, tminmax):

    windsock_weak = wind_sock(rot=50)
    windsock_medi = wind_sock(rot=70)
    windsock_stro = wind_sock(rot=90)

    q0, q1, q2 = storm_strength  # for readability

    y0, y1 = ax.get_ylim()

    ax.scatter(
        [d for q, d in zip(q0, dates) if q],
        np.ones_like(dates)[q0] * y0 + (y1 - y0) * 0.04,
        300,
        color=p_storm[q0, :],
        marker=windsock_weak,
    )
    ax.scatter(
        [d for q, d in zip(q1, dates) if q],
        np.ones_like(dates)[q1] * y0 + (y1 - y0) * 0.04,
        350,
        color=p_storm[q1, :],
        marker=windsock_medi,
    )
    ax.scatter(
        [d for q, d in zip(q2, dates) if q],
        np.ones_like(dates)[q2] * y0 + (y1 - y0) * 0.04,
        400,
        color=p_storm[q2, :],
        marker=windsock_stro,
    )


def lightning_plotter(ax, dates, p_light):
    boltpath = lightning_bolt()
    ax.scatter(dates, 0.1 * np.ones_like(dates), 80, marker=boltpath, color=p_light, zorder=5)


def prob_meteogram(
    loc: Loc, data,  # or maybe separate lat,lon,address kwargs would be better here
):
    """
    Create *probabilistic meteogram*.


    """
    # create figure and axes
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 6], hspace=0, figure=fig)
    cloud_ax = fig.add_subplot(gs[0])
    rain_ax = fig.add_subplot(gs[1])
    temp_ax = fig.add_subplot(gs[2])

    fig.tight_layout(rect=[0.02, 0.03, 1, 0.97])
    # TODO: try use constrained_layout instead?

    # grab needed data variables and time coordinate variable
    # it is assumed that the data dimensions are (time, ensemble)
    time_utc = data["time_utc"]
    t = data["t"]
    lcc = data["lcc"]
    mcc = data["mcc"]
    hcc = data["hcc"]
    lsp = data["lsp"]
    u = data["u"]
    v = data["v"]

    # shift dates according to timezone
    utcoffset = timezone_offset(loc, time_utc[0])
    time = [_ + utcoffset for _ in time_utc]

    # shifted datevector for rain
    three_hours = datetime.timedelta(hours=3)
    rtime = [_ + three_hours for _ in time[:-1]]

    # temperature
    t = np.sort(t)  # sort temperature by ensemble members
    # ? why is ^ needed?
    tmedian = np.median(t, axis=1)  # ensemble median
    tminmax = (t.min(), t.max())  # used for axis formatting

    # interpolate data
    time_num = date2num(time)  # original time values in number format
    time_interp = spline_dates(time)  # new time values as datetimes
    t_interp = spline_data_by_date(t, time_num)
    lcc_interp = spline_data_by_date(lcc, time_num)
    mcc_interp = spline_data_by_date(mcc, time_num)
    hcc_interp = spline_data_by_date(hcc, time_num)

    # rain
    lightrain, medrain, heavyrain, rain_explanation = rain_prob(lsp, rtime)

    # (horizontal) wind speed
    spd = np.sqrt(u ** 2 + v ** 2)

    # storminess
    p_storm, storm_strength = storm(spd)

    # lightning chance (rough estimate)
    p_light = lightning(lcc, mcc, hcc, tmedian)

    # snow (very rough estimate) (more like frozen/liquid precip)
    snow = snow_or_rain(tmedian)

    # do axes formatting
    cloud_ax_format(cloud_ax, time, loc)
    rain_ax_format(rain_ax, time, rain_explanation, snow)
    temp_ax_format(temp_ax, tminmax, time, utcoffset, loc)

    # plotters
    temp_plotter(temp_ax, time, t_interp, tminmax)
    clouds_plotter(cloud_ax, time, hcc_interp, mcc_interp, lcc_interp)
    rain_plotter(rain_ax, lightrain, medrain, heavyrain, snow, rtime)
    wind_plotter(temp_ax, time, p_storm, storm_strength, tminmax)
    lightning_plotter(cloud_ax, time, p_light)


if __name__ == "__main__":
    pass
