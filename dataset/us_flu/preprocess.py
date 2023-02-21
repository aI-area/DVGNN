import pandas as pd
import numpy as np

LENGTH = 53  # length of each data series
START_YEAR = 2003
START_WEEK = 41
END_YEAR = 2019
END_WEEK = 52

df = pd.read_csv(
    'ILINet.csv',
    skiprows=1
)
df['ILI RATE'] = df['ILITOTAL'] / df['TOTAL PATIENTS']  # calculate ILI rates

# select data in the time period
start_index = df[(df['YEAR'] == START_YEAR) & (df['WEEK'] == START_WEEK)].index[0] - LENGTH + 1
end_index = df[(df['YEAR'] == END_YEAR) & (df['WEEK'] == END_WEEK)].index[0]
df = df.loc[start_index:end_index, ['YEAR', 'WEEK', 'ILI RATE']].reset_index(drop=True)

# slide window
series_data = pd.concat(
    [df.loc[i-LENGTH+1:i, 'ILI RATE'].reset_index(drop=True)
     for i in range(LENGTH-1, df.shape[0])],
    axis=1
).T.reset_index(drop=True)
series_data.columns = [f'wk_{i+1}' for i in range(LENGTH)]
time_data = df.loc[LENGTH-1:, ['YEAR', 'WEEK']].reset_index(drop=True)
data = pd.concat([time_data, series_data], axis=1)

# save data
data.to_csv('us_ratio.csv', header=True, index=True)






