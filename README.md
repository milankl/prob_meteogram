# Intuitive probabilistic meteograms

This is a repository intended to develop intuitive probabilistic meteograms for an increased public confidence in weather forecasts. It is based on the ensemble members of an ensemble prediction system and aims to visualize to weather at a given location in a way that is easy to understand for everybody.

# Problem
A 7-day weather forecasts is now as accurate as a 5-day forecast 20 years ago, thanks to faster supercomputers and decades of scientific weather forecast model development. However, there is still a widespread public misbelief that forecasts are not reliable. Traditional meteograms (i.e. a plot summarizing the weather for a certain location throughout the next days or week) do not contain any information about the uncertainty of a weather forecast. A paradigm exists where a layperson is not expected to understand the uncertainty of a forecast. Especially weather apps tend to present forecasts in an overly simplistic way that may support the aforementioned little public confidence in weather forecasting.

Example: Every forecast of an ensemble forecast might propose a little bit of rain in the mornings or evenings over the coming weekend. The overly simplistic way of communicating this forecast as "Rain throughout Saturday and Sunday" gives one the impression of an unsuitable weekend for camping. In contrast, the information that in each forecast it actually rained <5% of the time is not well communicated and could have been presented in an intuitive way, understandable for a layperson.

# Proposed solution

The solution I propose is an inuitive way of visualizing the probabilistic information of an ensemble forecast in the form of a meteogram (as the public is mostly interested in the weather at a given location). By "intuitive" I mean a meteogram that is easy to understand for everyone without a statistics background. It is supposed to present the future weather at a given location in a very condensed way, such that 10 seconds are enough to understand what you are likely to expect and what the possible extremes are. I believe that the additional information about the uncertainty of a forecast will increase public confidence in weather forecasting, important for a wider outreach of publicly funded data.

# Dependencies

    Geopy, tzwhere

can be installed via pip.

# HOW TO USE

This script will create a probabilistc meteogram for a given city or location. Will require ECMWF EPS data in a folder called "/data", which is not provided in this repository. Then do

    python meteogram.py
    
to create a meteogram for Rio de Janeiro, Brazil. You can also to

    python meteogram.py "ECMWF Reading"

and specify your own location.