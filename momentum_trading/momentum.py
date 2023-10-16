import pandas as pd
import numpy as np

# Read data
crsp = pd.read_csv('file_path')

# Convert the Observation date into a datetime format and extract year and month
crsp['date'] = pd.to_datetime(crsp['DateOfObservation'].astype(str), format='%Y%m%d')
crsp['year'] = crsp['date'].dt.year
crsp['month'] = crsp['date'].dt.month

# Define the getMomentum function
def getMomentum(thisPermno, thisYear, thisMonth, crsp):
    if thisMonth == 1:
        t_1_month, t_1_year = 12, thisYear - 1
        t_12_month, t_12_year = 1, thisYear - 1
    else:
        t_1_month, t_1_year = thisMonth - 1, thisYear
        t_12_month, t_12_year = thisMonth, thisYear - 1

    price_t1 = crsp.loc[(crsp['PERMNO'] == thisPermno) & (crsp['year'] == t_1_year) & (crsp['month'] == t_1_month), 'adjustedPrice']
    price_t12 = crsp.loc[(crsp['PERMNO'] == thisPermno) & (crsp['year'] == t_12_year) & (crsp['month'] == t_12_month), 'adjustedPrice']

    if price_t1.empty or price_t12.empty:
        return np.nan
    else:
        return price_t1.values[0] / price_t12.values[0]

# Calculate the continuous return (momentum)
crsp['momentum'] = [getMomentum(crsp['PERMNO'].iloc[i], crsp['year'].iloc[i], crsp['month'].iloc[i], crsp) for i in range(len(crsp))]

# Create a new dataframe for momentum with unique dates
momentum_tbl = pd.DataFrame({'unique_dates': crsp['DateOfObservation'].unique()})
momentum_tbl['date'] = pd.to_datetime(momentum_tbl['unique_dates'].astype(str), format='%Y%m%d')
momentum_tbl['year'], momentum_tbl['month'] = momentum_tbl['date'].dt.year, momentum_tbl['date'].dt.month

momentum_tbl['mom1'] = np.nan
momentum_tbl['mom10'] = np.nan
momentum_tbl['mom'] = np.nan

for i in range(len(momentum_tbl)):
    cur_year = momentum_tbl['year'].iloc[i]
    cur_month = momentum_tbl['month'].iloc[i]

    filtered_crsp = crsp[(crsp['year'] == cur_year) & (crsp['month'] == cur_month)]
    mom_values = filtered_crsp['momentum'].dropna()

    if not mom_values.empty:
        q = mom_values.quantile([0.1, 0.9])
        loser_tickers = filtered_crsp['PERMNO'][mom_values <= q[0.1]]
        winner_tickers = filtered_crsp['PERMNO'][mom_values >= q[0.9]]

        loser_returns = filtered_crsp['Returns'][filtered_crsp['PERMNO'].isin(loser_tickers)]
        winner_returns = filtered_crsp['Returns'][filtered_crsp['PERMNO'].isin(winner_tickers)]

        momentum_tbl.at[i, 'mom1'] = loser_returns.mean(skipna=True)
        momentum_tbl.at[i, 'mom10'] = winner_returns.mean(skipna=True)
        momentum_tbl.at[i, 'mom'] = winner_returns.mean(skipna=True) - loser_returns.mean(skipna=True)

momentum_tbl['cumulativeRet'] = (1 + momentum_tbl['mom']).cumprod() - 1
momentum_tbl['cumulativeRet'].fillna(0, inplace=True)

