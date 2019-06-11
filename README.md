# Intuitive probabilistic meteograms
![meteogram](examples/meteogram_Alderaan%20City.png?raw=true "Meteogram")

This is a repository intended to develop intuitive probabilistic meteograms for an increased public confidence in weather forecasts. It is based on the ensemble members of an ensemble prediction system and aims to visualize to weather at a given location in a way that is easy to understand for everybody.

# Problem

A 7-day weather forecasts is now as accurate as a 5-day forecast 20 years ago, thanks to faster supercomputers and decades of scientific weather forecast model development. However, there is still a widespread public misbelief that forecasts are not reliable. Traditional meteograms, summarizing the weather for a certain location throughout the next week, rarely contain information about the uncertainty of a weather forecast. A paradigm exists where a layperson is not expected to understand the uncertainty of a forecast. Especially smartphone weather apps tend to present forecasts in an overly simplistic way that may support the aforementioned little public confidence in weather forecasting.

# Proposed solution

The proposed solution is an intuitive way of visualizing the probabilistic information of an ensemble forecast in the form of a meteogram, as the public is mostly interested in the weather at a given location. The meteogram is designed to be easily understandable for everyone without a statistics background. The temperature is plotted as a colour-shaded time series, where width and transparency denote the uncertainty and colour is used to provide an intuitive feeling for the temperature (blue=cold, red=warm). Strong winds can be seen on the bottom of that panel, where a windsock indicates the expected strength. Transparency denotes the uncertainty of the wind forecast. The upper panels provide information about cloudiness and precipitation. Rain and snow are split into three categories: light, medium and heavy that are displayed in three rows, each with transparency to denote the uncertainty of a rainfall event at a given strength. In case of snow, the rain droplets are changed into snow flakes. The precipitation panel therefore can display events like “very likely weak rain, but unlikely stronger” or “equally likely medium or heavy rain”. Cloudiness is presented in the top panel, indicating low, medium and high clouds in different shades of grey. The thickness represents the cloud cover of each cloud type. Additionally, a lightning bolt may indicate once the chance of thunder is beyond a given threshold. Sunrise and sunset times are provided in the temperature panel.

# Alderaan City example

The weather forecast for Alderaan City shows warm temperatures during the day around 20C for the next days, although nights are cold. Weak rain and thunder has to be expected on Sunday. Monday will be the most beautiful day of the week with low cloud cover and temperatures around 14C. During the week, temperatures are expected to drop and uncertainty increases towards next weekend. Strong winds have to be expected for Mon, Tue and Wed, that will peak on Wednesday noon. Probably strong rainfall from Tue onwards, increasing in likelihood and strength towards Thu and Fri. Full cloud cover throughout the week, but thinner on the next weekend.

We believe that the additional information about the uncertainty of a forecast will increase public confidence in weather forecasting, important for a wider outreach of publicly funded data.

# Dependencies

    Geopy, tzwhere

can be installed via pip.

# HOW TO USE

This script will create a probabilistc meteogram for a given city or location. Will require ECMWF EPS data in a folder called "/data", which is not provided in this repository due to size and licensing issues. However, you can exectue

    python meteogram_alderaan.py
    
which creates a fake meteogram based on the available .npz files in data/. In case EPS data is available in the folder you can simply do

    python meteogram.py "ECMWF Reading" 1

and specify your own location and whether you want an .png output with 1 or 0.
