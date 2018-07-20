import os, sys, inspect
HERE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(HERE_PATH)

locations = ["Santiago Chile",
            "Kingston Jamaica",
            "Oxford UK",
            "Hobart Australia",
            "Fairbanks Alaska",
            "Cape Town South Africa",
            "Rio de Janeiro Brazil",
            "Lisbon Portugal",
            "Honolulu Hawaii",
            "Hanoi Vietnam",
            "Moscow Russia",
            "Delhi India",
            "Rome Italy",
            "Accra Ghana",
            "Teheran Iran",
            "Christchurch New Zealand"]
            
# locations = ["Istanbul Turkey",
#             "Vancouver Canada",
#             "San Francisco USA",
#             "Chicago USA",
#             "Miami USA",
#             "Lima Peru",
#             "New York USA",
#             "Reykjavik Iceland",
#             "Nairobi Kenia",
#             "Astana Kasachstan",
#             "Kathmandu Nepal",
#             "Seoul South Korea",
#             "Yakutsk Russia",
#             "Syndey Australia",
#             "Perth Australia",
#             "Auckland New Zealand",
#             "Stockholm Sweden",
#             "Copenhagen Denmark",
#             "Hamburg Germany",
#             "Prague Czech Republic"]

for loc in locations:
    os.system("python meteogram.py '"+loc+"' 1")
    print("Meteogram created for "+loc)