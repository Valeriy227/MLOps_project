import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if len(sys.argv) != 3:
    print('Usage: python3 {} <FILE-IN> <FILE-OUT>'.format(sys.argv[0]))
    print('Wrong number of command-line arguments')
    sys.exit(1)

path_in = sys.argv[1]
path_out = sys.argv[2]
df = pd.read_parquet(path_in)

print('Length of dataframe is {}'.format(len(df)))

print('Column names')
print(df.columns)

print('Column types')
print(df.dtypes)

print('Dropping total_amount and tolls_amount (they include the target)')
df.drop(columns=['total_amount', 'tolls_amount'], inplace=True)

for col in df.columns:
    print('#unique in {} is {}'.format(col, df[col].nunique()))

for col in df.columns:
    print('percentage of NAs in {} is {:.1f}%'.format(col, 100. * df[col].isna().sum() / len(df)))

print('Droping rows with missing values')
df.dropna(inplace=True)
print('Now length of dataframe is {}'.format(len(df)))

print('Convert Store_and_fwd_flag to number')
df.store_and_fwd_flag = (df.store_and_fwd_flag == 'Y').astype('int64')

for col in df.columns:
    print('Droping rows with anomalous values of {}'.format(col))
    lower_bound = df[col].quantile(0.005)
    upper_bound = df[col].quantile(0.995)

    cond = (df[col] < lower_bound) | (df[col] > upper_bound)
    df.drop(df[cond].index, inplace=True)

    print('Now length of dataframe is {}'.format(len(df)))

target = np.log(df.tip_amount + 1.)
plt.hist(target)
plt.title('Распределение логарифма чаевых')
plt.xlabel('log_2(tip_amount + 1)')
plt.ylabel('Частота')
plt.savefig('target_dist.png')

print('Dropping tip_amount from dataframe')
df.drop(columns=['tip_amount'], inplace=True)

df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime) / datetime.timedelta(minutes=1)
print('Dropping tpep_pickup_datetime and tpep_dropoff_datetime')
df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], inplace=True)

cols_to_drop = []
for col in df.columns:
    if df[col].nunique() < 300:
        print('Variable {} is categorial, doing one hot encoding of it'.format(col))
        cols_to_drop.append(col)

        for value in df[col].unique():
            new_col_name = '{}=={}'.format(col, value)
            new_col_val = (df[col] == value).astype('int64')
            c = target.corr(new_col_val)
            if (not np.isnan(c)) and abs(c) > 0.015:
                df[new_col_name] = new_col_val  

df.drop(columns=cols_to_drop, inplace=True)

print('Correlations with target:')
for col in df.columns:
    c = target.corr(df[col])
    print('{} {:.2f}'.format(col, c))

result = pd.DataFrame()
result['target'] = target

for i, col in enumerate(df.columns):
    mu = df[col].mean()
    sigma = df[col].std()
    result[str(i)] = (df[col] - mu) / sigma

result.to_parquet(path_out)
