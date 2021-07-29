'''
  Compare charts to other charts
'''
import _config, util
import sys, os, pickle, fnmatch, datetime, subprocess, functools, re
import numpy as np, pandas as pd
from collections import defaultdict, Counter

import _qsub, _stepcharts
import chart_tag

# Default params
inp_dir_a = _config.OUT_PLACE + 'a_format_data/'
inp_dir_d = _config.OUT_PLACE + 'd_annotate/'
inp_dir_merge = _config.OUT_PLACE + 'merge_features/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

scinfo = _stepcharts.SCInfo()

CONTEXT_LOWER = -3
CONTEXT_UPPER = 0


'''
  Context
'''
def get_context_df(nm):
  level = scinfo.name_to_level[nm]
  df = pd.read_csv(inp_dir_merge + f'features.csv', index_col=0)
  df = df.drop_duplicates(subset = 'Name (unique)', keep='last')

  # Add local
  use_local = False
  # use_local = True
  if use_local:
    local_fn = inp_dir_d + f'{nm}_features.csv'
    if os.path.isfile(local_fn):
      print('Using local file ...')
      local_df = pd.read_csv(local_fn, index_col=0)
      df = df.append(local_df)

  steptype = df[df['Name (unique)'] == nm]['Steptype simple']
  if 'S' in steptype:
    df = _stepcharts.singles_subset(df)
  elif 'D' in steptype:
    df = _stepcharts.doubles_subset(df)

  context_crit = (df['Level'] >= level + CONTEXT_LOWER) & \
                 (df['Level'] <= level + CONTEXT_UPPER)
  context_df = df[context_crit]
  return context_df


'''
  Compare charts by tech percentiles
'''

def get_custom_timelines(tag_percentiles, n = 3, verbose = True):
  exclude = [
    'Hold run',
    'Run',
    'Hold',
    'Hold taps',
    'Twist angle - none',
    'Twist angle - 90',
    'Twist angle - close diagonal',
    'Irregular rhythm',
    'Rhythm change',
  ]
  found_tags = {k: v for k, v in tag_percentiles.items() if k not in exclude}

  tags = sorted(found_tags, key=found_tags.get, reverse=True)
  if verbose:
    for tag in tags:
      print(tag.ljust(30), f'{found_tags[tag]:.1%}')
  return tags[:n]


def get_tech_percentiles(context_df, nm):
  '''
    Used for picking custom timelines and finding similar charts
  '''
  kw = ' - 50% nps'
  broad_tags = [col.replace(kw, '') for col in context_df.columns if kw in col]

  # Zero out percentiles when frequency is below threshold
  LOW_FQ_THRESHOLD = 0.05

  row = context_df[context_df['Name (unique)'] == nm].iloc[0]
  tag_percentiles = {}
  for tag in broad_tags:
    col = f'{tag} - frequency'
    fq, context = row[col], context_df[col]
    pct = sum(context < fq) / len(context)
    if fq > LOW_FQ_THRESHOLD:
      tag_percentiles[tag] = pct
    else:
      tag_percentiles[tag] = 0
  return tag_percentiles


'''
  Rename tags
'''
def rename_tag(t):
  annots = {
    'Staggered hit': 'Rolling hit',
    'Hold tap single foot': 'Bracket hold tap',
    'Hold footslide': 'Hold f.slide',
    'Hold footswitch': 'Hold f.switch',
    'Stairs, singles': '5 stair',
    'Stairs, doubles': '10 stair',
    'Broken stairs, doubles': '9 stair',
    'Twist angle - 180': '180° twist',
    'Twist angle - far diagonal': 'Diag. twist',
    'Twist angle - close diagonal': 'Diag. twist',
    'Twist solo diagonal': 'Diag. twist',
    'Twist angle - 90': '90° twist',
  }
  return annots.get(t, t)


'''
  Run
'''
def run_single(nm, verbose = False):
  '''
    Compares a stepchart to other stepcharts of a similar level
    Finds technique tags and descriptions of stepchart
  '''
  context_df = get_context_df(nm)

  tags = chart_tag.tag_chart(context_df, nm)
  if verbose:
    print('\n', nm)
    for k, v in tags.items():
      print(k.ljust(20), v)

  tag_pcts = get_tech_percentiles(context_df, nm)
  timeline_tags = get_custom_timelines(tag_pcts, n=3, verbose=verbose)
  desc = ''

  tag_d = {
    'description': desc,
    'timeline tags': timeline_tags,
  } 
  return tag_d


@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  # nm = 'Super Fantasy - SHK S16 arcade'
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
  nm = 'Rock the house - Matduke D22 arcade'

  run_single(nm, verbose=True)
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