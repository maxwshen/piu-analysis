'''
    Tag charts
'''
import _config, util
import sys, os, pickle, fnmatch, datetime, subprocess, functools, re
import numpy as np, pandas as pd
from collections import defaultdict, Counter

import _qsub, _stepcharts

# Default params
inp_dir_a = _config.OUT_PLACE + 'a_format_data/'
inp_dir_d = _config.OUT_PLACE + 'd_annotate/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

scinfo = _stepcharts.SCInfo()

CONTEXT_LOWER = -3
CONTEXT_UPPER = 0


'''
'''
def tag_chart(context_df, nm):
  row = context_df[context_df['Name (unique)'] == nm].iloc[0]
  found_tags = {}
  for tag in get_tags(context_df):
    keep, ranker = get_stats(row, context_df, tag)
    if keep:
      found_tags[tag] = ranker

  tags = sorted(found_tags, key=found_tags.get, reverse=True)

  tag_to_adjs = dict()
  for tag in tags:
    tag_to_adjs[tag] = get_adjectives(row, context_df, tag)

  return tag_to_adjs


def get_tags(df):
  kw = ' - 50% nps'
  tags = [col.replace(kw, '') for col in df.columns if kw in col]
  exclude = {
    'Twist angle - none',
    'Twist solo diagonal',
    'Twist angle - 90',
    'Twist angle - close diagonal',
    'Twist angle - far diagonal',
    # 'Twist angle - 180',
  }
  return [tag for tag in tags if tag not in exclude]


MIN_INTERESTING_NPS = 5   # todo - level dependent

def get_stats(row, context_df, tag, verbose = True):
  '''
    Include tag if (OR)
    - Frequency is in top 80% percentile of context charts
    - Frequency is above 10%
  '''
  PCT_THRESHOLD = 0.80
  OBJECTIVE_MIN_FQ = 0.10
  suffix_to_adjective = {
    ' - frequency': '',
  }
  col = f'{tag} - frequency'
  val, context = row[col], context_df[col]
  pct = sum(context < val) / len(context)

  keep = False
  ranker = 0
  if pct >= PCT_THRESHOLD or val >= OBJECTIVE_MIN_FQ:
    keep = True
    ranker = pct

  if verbose:
    print(col.ljust(30), f'{val:.2f} {pct:.0%}')
  return keep, ranker


def get_adjectives(row, context_df, tag, verbose = True):
  '''
    Include these adjectives only for tags included for other reasons
  '''
  adjs = dict()
  adjs.update(twistiness(row, context_df, tag, verbose))
  adjs.update(speed(row, context_df, tag, verbose))
  adjs.update(length(row, context_df, tag, verbose))

  adj_kws = sorted(adjs, key=adjs.get, reverse=True)
  return adj_kws


def speed(row, context_df, tag, verbose):
  '''
    Label 'fast' only if (AND)
    - speed (80% nps or nps of longest) is above average within chart
    - speed is high percentile among context charts
    - speed is above a minimum interesting nps
    - frequency of movement pattern is in top 50th percentile
  '''
  speed_cols = [
    # ' - 80% nps',
    ' - median nps of longest'
  ]

  fq_col = f'{tag} - frequency'
  fq = row[fq_col]
  # fq_pct = sum(context_df[fq_col] < fq) / len(context_df)

  adjs = dict()
  for suffix in speed_cols:
    col = f'{tag}{suffix}'
    if col in row.index:
      val, context = row[col], context_df[col]
      pct = sum(context < val) / len(context)
      nps_str = f'{val:.1f} nps'
      adjs[nps_str] = pct

      ebpm_str = effective_bpm(val, row['BPM mode'])
      adjs[ebpm_str] = pct
      # print(ebpm_str)
      # import code; code.interact(local=dict(globals(), **locals()))

      if verbose:
        print(col.ljust(30), f'{val:.2f} {pct:.0%}')
  return adjs


def effective_bpm(nps, base_bpm):
  npm = nps * 60

  # Get lower and upper bpm, ensure that 2x does not fit in
  lower = base_bpm * 2/3
  upper = base_bpm * 4/3
  while lower * 2 < upper:
    lower += 1
  
  factors = {
    4:    'whole note',
    2:    'half note',
    1:    'quarter note',
    1/2:  '8th note',
    # 1/3:  '12th note',
    1/4:  '16th note',
    # 1/6:  '24th note',
    1/8:  '32nd note',
    # 1/12: '48th note',
    1/16: '64th note',
  }
  for factor in factors:
    ebpm = npm * factor
    if lower <= ebpm <= upper:
      break
  note_type = factors[factor]
  ebpm_str = f'{note_type}s at {ebpm:.0f} bpm'
  return ebpm_str


def length(row, context_df, tag, verbose):
  # Add 'long' using max len sec and nps of longest
  PCT_THRESHOLD = 0.80
  mean_nps = row['Notes per second since downpress - mean']
  adjs = dict()
  col = f'{tag} - max len sec'
  if col in row.index:
    val, context = row[col], context_df[col]
    pct = sum(context < val) / len(context)
    nps = row[f'{tag} - mean nps of longest']
    if pct >= PCT_THRESHOLD and nps >= MIN_INTERESTING_NPS and nps >= mean_nps:
      adjs['long'] = pct
  return adjs


def twistiness(row, context_df, tag, verbose):
  PCT_THRESHOLD = 0.65
  suffix_to_adjective = {
    ' - % no twist': 'front-facing',
    ' - % 90+ twist': 'twisty',
    ' - % diagonal+ twist': 'diagonal twisty',
    ' - % far diagonal+ twist': 'has hard diagonal twists',
    ' - mean travel (mm)': 'travel',
    ' - 80% travel (mm)': 'travel',
  }

  adjs = dict()
  for suffix, adjective in suffix_to_adjective.items():
    col = f'{tag}{suffix}'
    if col in row.index:
      val, context = row[col], context_df[col]
      pct = sum(context < val) / len(context)
      if pct >= PCT_THRESHOLD:
        adjs[adjective] = pct
      if verbose:
        print(col.ljust(30), f'{val:.2f} {pct:.0%}', )
  return adjs


'''
  Run
'''
def run_single(nm):
  level = scinfo.name_to_level[nm]
  df = pd.read_csv(_config.OUT_PLACE + f'features.csv', index_col=0)
  df = df.fillna(0)

  # Add local
  use_local = False
  # use_local = True
  if use_local:
    local_fn = inp_dir_d + f'{nm}_features.csv'
    if os.path.isfile(local_fn):
      print('Using local file ...')
      local_df = pd.read_csv(local_fn, index_col=0)
      df = df.append(local_df)

  df['Name (unique)'] = df.index
  df = df.drop_duplicates(subset = 'Name (unique)', keep='last')

  # Annotate
  all_df = pd.read_csv(inp_dir_a + 'all_stepcharts.csv', index_col=0)
  all_df['Level'] = all_df['METER']
  df = df.merge(all_df, on = 'Name (unique)', how = 'left')

  steptype = df[df['Name (unique)'] == nm]['Steptype simple']
  if 'S' in steptype:
    df = _stepcharts.singles_subset(df)
  elif 'D' in steptype:
    df = _stepcharts.doubles_subset(df)

  context_crit = (df['Level'] >= level + CONTEXT_LOWER) & \
                 (df['Level'] <= level + CONTEXT_UPPER)
  context_df = df[context_crit]

  tags = tag_chart(context_df, nm)

  print('\n', nm)
  for k, v in tags.items():
    print(k.ljust(20), v)
  return


@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  # nm = 'Super Fantasy - SHK S16 arcade'
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
  nm = 'Mitotsudaira - ETIA. D19 arcade'

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