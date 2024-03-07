import pandas as pd

# import
data = pd.read_csv("country_vaccination_stats.csv")

#  date format
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

# fill wtih min daily vacation number of country
data['daily_vaccinations'] = data.groupby('country')['daily_vaccinations'].transform(lambda x: x.fillna(x.min()))

# if missed data exist then replace null w 0
data.fillna(0, inplace=True)

# daily vacation median for all country
top_3_countries = data.groupby('country')['daily_vaccinations'].median().nlargest(3)

print("Top 3 countries with the highest median daily vaccination numbers:")
print(top_3_countries)
