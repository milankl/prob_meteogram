"""
CLI
"""
import sys
import warnings

from geopy.geocoders import Nominatim

from .core import LOC_DEFAULT


if len(sys.argv) > 1:
    LOC_ARG = sys.argv[1]
    OUTPUT = sys.argv[2]
else:
    LOC_ARG = "Rio de Janeiro Brazil"
    OUTPUT = 0

geolocator = Nominatim()
try:
    loc = geolocator.geocode(LOC_ARG)
except:  # no internet connection or server request failed
    loc = LOC_DEFAULT
    warnings.warn(f"Geolocation failed. Using {loc.address} instead.")

