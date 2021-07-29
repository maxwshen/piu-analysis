'''
  Featurize charts for clustering
'''
import _config, util
import sys, os, pickle, fnmatch, datetime, subprocess, functools, re
import numpy as np, pandas as pd
from collections import defaultdict, Counter

import _qsub
import chart_compare

# Default params
inp_dir_merge = _config.OUT_PLACE + 'merge_features/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

all_df = pd.read_csv(inp_dir_merge + 'features.csv', index_col=0)

'''
  Run
'''
def run_multiple(start, end):
  mdf = pd.DataFrame()
  timer = util.Timer(total=end-start)
  for i, row in all_df.iloc[start:end].iterrows():
    nm = row['Name (unique)']
    df = run_single(nm, save=False)
    mdf = mdf.append(df, ignore_index=True)
    timer.update()
  mdf.to_csv(out_dir + f'merged_{start}_{end}.csv')
  return


def run_single(nm, save = True):
  context_df = chart_compare.get_context_df(nm)
  tag_pcts = chart_compare.get_tech_percentiles(context_df, nm)

  df = pd.DataFrame(tag_pcts, index=[nm])
  if save:
    df.to_csv(out_dir + f'{nm}.csv')
  return df


def merge_all():
  mdf = pd.DataFrame()
  timer = util.Timer(total=len(all_df))
  for i, row in all_df.iterrows():
    nm = row['Name (unique)']
    fn = out_dir + f'{nm}.csv'
    if os.path.isfile(fn):
      df = pd.read_csv(fn, index_col=0)
      mdf = mdf.append(df)
    timer.update()
  print(f'Merged {len(mdf)} files.')

  mdf = mdf.rename(columns = {c: f'Feature - {c}' for c in mdf.columns})

  mdf['Name (unique)'] = mdf.index
  dfs = all_df[['Name (unique)', 'Level', 'Is singles', 'Is doubles']]
  mdf = mdf.merge(dfs, on='Name (unique)')

  mdf.to_csv(out_dir + f'cluster_features.csv')
  return


@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  nm = 'Super Fantasy - SHK S16 arcade'
  # nm = 'Nakakapagpabagabag - Dasu feat. Kagamine Len S18 arcade'
  # nm = 'Super Fantasy - SHK S19 arcade'
  # nm = 'Native - SHK S20 arcade'
  # nm = 'Mr. Larpus - BanYa S22 arcade'
  # nm = 'Conflict - Siromaru + Cranky S17 arcade'
  # nm = 'Bad End Night - HitoshizukuP x yama S17 arcade'
  # nm = 'Gothique Resonance - P4Koo S20 arcade'
  # nm = 'Sorceress Elise - YAHPP S23 arcade'
  # nm = 'U Got 2 Know - MAX S20 arcade'
  # nm = 'YOU AND I - Dreamcatcher S21 arcade'
  # nm = 'Death Moon - SHK S22 shortcut'
  # nm = 'King of Sales - Norazo S21 arcade'
  # nm = 'Tepris - Doin S17 arcade'
  # nm = '8 6 - DASU S20 arcade'
  # nm = 'The End of the World ft. Skizzo - MonstDeath S20 arcade'
  # nm = 'Bad Apple!! feat. Nomico - Masayoshi Minoshima S17 arcade'

  # Doubles
  # nm = 'Mitotsudaira - ETIA. D19 arcade'
  # nm = 'Rock the house - Matduke D22 arcade'

  run_single(nm)
  return


if __name__ == '__main__':
  if len(sys.argv) == 1:
    main()
  else:
    if sys.argv[1] == 'gen_qsubs':
      _qsub.gen_qsubs(NAME, sys.argv[2])
    elif sys.argv[1] == 'run_qsubs':
      _qsub.run_qsubs(
        chart_fnm = sys.argv[2],
        start = sys.argv[3],
        end = sys.argv[4],
        run_single = run_single,
      )
    elif sys.argv[1] == 'gen_qsubs_remainder':
      _qsub.gen_qsubs_remainder(NAME, sys.argv[2], '.csv')
    elif sys.argv[1] == 'run_single':
      run_single(sys.argv[2])
    elif sys.argv[1] == 'merge_all':
      merge_all()