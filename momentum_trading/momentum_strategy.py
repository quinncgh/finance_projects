import pandas as pd
import numpy as np

# Import the data
crsp = pd.read_csv('testData.csv')

# Convert the Observation date into string, then create new columns for month and year
crsp['datenum'] = pd.to_datetime(crsp['DateOfObservation'].astype(str))
crsp['year'] = crsp['datenum'].dt.year
crsp['month'] = crsp['datenum'].dt.month

def getMomentum(thisPermno, thisYear, thisMonth, crsp_df):
    if thisMonth == 1:
        t_1_month = 12
        t_1_year = thisYear - 1
        t_12_month = 1
        t_12_year = thisYear -1
    else:
        t_1_month = thisMonth -1
        t_1_year = thisYear
        t_12_month = thisMonth
        t_12_year = thisYear -1

    price_t1 = crsp_df.loc[(crsp_df['PERMNO'] == thisPermno) & (crsp_df['year'] == t_1_year) & (crsp_df['month'] == t_1_month), 'adjustedPrice']
    price_t12 = crsp_df.loc[(crsp_df['PERMNO'] == thisPermno) & (crsp_df['year'] == t_12_year) & (crsp_df['month'] == t_12_month), 'adjustedPrice']

    if price_t1.empty or price_t12.empty:
        return np.nan
    else:
        return price_t1.iloc[0] / price_t12.iloc[0]

# Calculate the continuous return (momentum)
crsp['momentum'] = [getMomentum(row['PERMNO'], row['year'], row['month'], crsp) for _, row in crsp.iterrows()]

# Create a new DataFrame for the momentum with unique dates
momentum_tbl = pd.DataFrame({'datenum': crsp['datenum'].unique()})
momentum_tbl['year'] = momentum_tbl['datenum'].dt.year
momentum_tbl['month'] = momentum_tbl['datenum'].dt.month

# Initialize columns to store the momentum
momentum_tbl['mom1'] = np.nan
momentum_tbl['mom10'] = np.nan
momentum_tbl['mom'] = np.nan

for _, row in momentum_tbl.iterrows():
    filtered_crsp = crsp[(crsp['year'] == row['year']) & (crsp['month'] == row['month']) & (~crsp['Returns'].isna())]
    q = filtered_crsp['momentum'].quantile([0.1, 0.9])

    short_returns = filtered_crsp.loc[filtered_crsp['PERMNO'].isin(filtered_crsp['PERMNO'][filtered_crsp['momentum'] <= q[0.1]]), 'Returns']
    long_returns = filtered_crsp.loc[filtered_crsp['PERMNO'].isin(filtered_crsp['PERMNO'][filtered_crsp['momentum'] >= q[0.9]]), 'Returns']

    row['mom1'] = short_returns.mean()
    row['mom10'] = long_returns.mean()
    row['mom'] = long_returns.mean() - short_returns.mean()

momentum_tbl['mom'].fillna(0, inplace=True)
momentum_tbl['cumulativeRet'] = np.nan
momentum_tbl.loc[0, 'cumulativeRet'] = 0
for i in range(1, len(momentum_tbl)):
    momentum_tbl.loc[i, 'cumulativeRet'] = (1 + momentum_tbl.loc[i, 'mom']) * (1 + momentum_tbl.loc[i-1, 'cumulativeRet']) - 1