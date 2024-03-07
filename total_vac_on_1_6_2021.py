import pandas as pd

# import
data = pd.read_csv("country_vaccination_stats.csv")

#  date format
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

# fill wtih min daily vacation number of country
data['daily_vaccinations'] = data.groupby('country')['daily_vaccinations'].transform(lambda x: x.fillna(x.min()))

# if missed data exist then replace null w 0
data.fillna(0, inplace=True)

# total vacation on 1/6/2021 
total_vaccinations_on_2021_01_06 = data[data['date'] == '2021-01-06']['daily_vaccinations'].sum()

print("Total Vacation # on 1/6/2021 :", total_vaccinations_on_2021_01_06)

