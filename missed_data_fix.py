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

# replace null w 0
data.fillna(0, inplace=True)

# check
print()
print(data.isnull().sum())
#done
