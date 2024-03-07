import pandas as pd

# import
data = pd.read_csv("country_vaccination_stats.csv")

print("dataset info")
print(data.head())
print()
print(data.describe())
print()
print(data.dtypes)
print()
# data col formatting
data['date'] = pd.to_datetime(data['date'], format='%m/%d/%Y')

print("CHECK MİSSED DATA")
print(data.isnull().sum())

# fill wtih min daily vacation number of country
for country in data['country'].unique():
    min_daily_vaccinations = data[data['country'] == country]['daily_vaccinations'].min()
    data.loc[data['country'] == country, 'daily_vaccinations'] = data.loc[data['country'] == country, 'daily_vaccinations'].fillna(min_daily_vaccinations)

# if missed data exist then replace null w 0
data.fillna(0, inplace=True)
print()
print(data)

# check
print()
print(data.isnull().sum())

#OK
