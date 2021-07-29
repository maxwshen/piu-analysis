'''
  Form data structure for front-end
'''
import _config, util
import sys, os, pickle, fnmatch, datetime, subprocess, functools, re
import numpy as np, pandas as pd
from collections import defaultdict, Counter

import _qsub, _stepcharts
import hmm_segment, e_struct_timelines, chart_compare

# Default params
inp_dir_a = _config.OUT_PLACE + 'a_format_data/'
inp_dir_d = _config.OUT_PLACE + 'd_annotate/'
inp_dir_merge = _config.OUT_PLACE + 'merge_features/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

scinfo = _stepcharts.SCInfo()

all_df = pd.read_csv(inp_dir_merge + 'features.csv', index_col=0)


'''
  Build data struct
'''
def chart_info(nm, compare_d):
  d = all_df[all_df['Name (unique)'] == nm].iloc[0]

  def expand_steptype(st):
    if 'S' in st:
      return 'Singles'
    elif 'D' in st:
      return 'Doubles'
    else:
      return 'Unknown'

  # Convert pred. difficulty pctile to text
  ranges = {
    (0, 0.10): ('Easy', '#7cb82f'),
    (0.10, 0.25): ('Easy-medium', '#efb920'),
    (0.25, 0.75): ('Medium', '#f47b16'),
    (0.75, 0.925): ('Hard', '#ec4339'),
    (0.925, 1.0): ('Very hard', '#c11f1d'),
  }
  dp = d['Predicted level percentile']
  dpkg = [pkg for r, pkg in ranges.items() if r[0] <= dp <= r[1]][0]

  chart_info_dict = {
    'name': nm,
    'song title': d['TITLE'],
    'artist': d['ARTIST'],
    'level': str(d['METER']),
    'song type': d['SONGTYPE'].capitalize(),
    'pack': d['Pack'], 
    'singles or doubles': expand_steptype(d['Steptype simple']),
    'description': compare_d['description'], 
    # 'related_charts': compare_d['related_charts'],
    # 'tags': [''], 
    'predicted difficulty': f'{d["Predicted level"]:.2f}',
    'difficulty string': dpkg[0],
    'difficulty string color': dpkg[1],
  }
  return convert_dict_to_js_lists(chart_info_dict)


def chart_card(nm, line_df, groups, compare_d):
  # Get xlabels based on group boundaries
  xticks = []
  xlabels = []
  times = list(line_df['Time'])
  for start, end in groups:
    t = times[end-1]
    xticks.append(t)
    min = round(t) // 60
    sec = round(t) % 60
    xlabels.append(f'{min}:{sec:02d}')

  # Construct timelines. Ordered: hold, twist, then 1-2-3 custom
  base_timelines = [
    e_struct_timelines.binary_timeline(line_df, 'Hold', 'Hold'),
    e_struct_timelines.twist(line_df),
  ]
  add_cols = compare_d['timeline tags']
  add_names = [chart_compare.rename_tag(t) for t in add_cols]
  add_timelines = [
    e_struct_timelines.binary_timeline(line_df, col, name)
    for col, name in zip(add_cols, add_names)
  ]

  chart_card_dict = {
    # 'groups': groups,
    'xticks': xticks,
    'xlabels': xlabels,
    'nps': e_struct_timelines.nps(line_df),
    'timelines': base_timelines + add_timelines,
  }
  return convert_dict_to_js_lists(chart_card_dict)


def chart_details(nm):
  chart_details_dict = {}
  return convert_dict_to_js_lists(chart_details_dict)


'''
  Support
'''
def convert_dict_to_js_lists(d):
  # Convert dict to [list of keys, list of values]
  vs = list(d.values())
  if any(type(v) == np.int64 for v in vs):
    print(f'Error: Found disallowed np.int64 type')
    raise Exception(f'Error: Found disallowed np.int64 type. Cast to int or str.')
  return [list(d.keys()), list(d.values())]


'''
  Run
'''
def run_single(nm):
  line_df = pd.read_csv(inp_dir_d + f'{nm}.csv', index_col=0)
  all_line_dfs, groups, comb_to_indiv = hmm_segment.segment(line_df)

  # anything that requires comparing chart to other charts (level-3 to level)
  compare_d = chart_compare.run_single(nm)

  struct = [
    chart_info(nm, compare_d),
    chart_card(nm, line_df, groups, compare_d),
    chart_details(nm),
  ]

  app_dir = f'../../piu-app/data/'
  with open(app_dir + f'{nm}.pkl', 'wb') as f:
    pickle.dump(struct, f)
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