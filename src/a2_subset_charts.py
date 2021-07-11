# 
import _config, _data, util, pickle
import sys, os, fnmatch, datetime, subprocess
import numpy as np
from collections import defaultdict, Counter
import pandas as pd

inp_dir = _config.OUT_PLACE + 'a_format_data/'
NAME = util.get_fn(__file__)
out_dir = _config.DATA_DIR
util.ensure_dir_exists(out_dir)

from functools import reduce

df = pd.read_csv(inp_dir + 'all_stepcharts.csv', index_col=0)
print('Total stepcharts:', len(df))

# unlabeled co-op charts
filter_charts = [
  'Witch Doctor - BanYa DP21 arcade infinity',
  'Cleaner - Doin DP17 arcade infinity',
  'Dream To Nightmare - Nightmare DP16 arcade infinity',
]

general_filters = [
  (df['METER'] != 99),
  (~df['Name (unique)'].isin(filter_charts)),
  (~df['Name (unique)'].str.contains('infinity')),
]

subsets = {
  'charts_singles': [
    (df['Steptype simple'].str.contains('S')),
  ],
  'charts_doubles': [
    (df['Steptype simple'].str.contains('D')),
  ],
  'charts_all': [],
  'charts_lowlevel': [(df['METER'] <= 15)],
  'charts_highlevel': [(df['METER'] > 15)],
}


def subset(subset_name):
  filters = general_filters + subsets[subset_name]
  dfs = [df[filt] for filt in filters]
  intersect = lambda df1, df2: pd.merge(df1, df2, how='inner')
  mdf = reduce(intersect, dfs)
  return mdf


@util.time_dec
def main():
  print(NAME)
  
  for subset_name in subsets:
    dfs = subset(subset_name)
    pct = len(dfs) / len(df)
    print(f'Created subset {subset_name}: {len(dfs)} stepcharts, {pct:.1%}')
    dfs.to_csv(out_dir + f'{subset_name}.csv')

  return


if __name__ == '__main__':
  main()