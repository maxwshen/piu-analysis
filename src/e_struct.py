'''
  Form data structure for front-end
'''
import _config, util
import sys, os, pickle, fnmatch, datetime, subprocess, functools, re
import numpy as np, pandas as pd
from collections import defaultdict, Counter

import _qsub, _stepcharts, _stances
import hmm_segment, e_struct_timelines, chart_compare, cluster_charts
import plot_chart

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
# 1
def chart_info(nm, compare_d):
  d = all_df[all_df['Name (unique)'] == nm].iloc[0]

  def expand_steptype(st):
    if 'S' in st:
      return 'Singles'
    elif 'D' in st:
      return 'Doubles'
    else:
      return 'Unknown'

  cluster_tags, neighbor_charts = cluster_charts.get_neighbors(nm)

  chart_info_dict = {
    'name': nm,
    'song title': d['TITLE'],
    'artist': d['ARTIST'],
    'level': str(d['METER']),
    'song type': d['SONGTYPE'].capitalize(),
    'pack': d['Pack'], 
    'singles or doubles': expand_steptype(d['Steptype simple']),
    'description': compare_d['description'], 
    'cluster tags': cluster_tags,
    'similar charts': neighbor_charts,
    # 'tags': [''], 
    'predicted difficulty': f'{d["Predicted level"]:.2f}',
    'predicted difficulty percentile': f'{d["Predicted level percentile"]:.2f}',
  }
  return chart_info_dict

# 2
def chart_card(nm, line_df, groups, compare_d, chart_info_dict):
  # Get xlabels based on group boundaries
  xticks = []
  xlabels = []
  times = list(line_df['Time'])
  for start, end in groups:
    t = times[end-1]
    xticks.append(t)
    xlabels.append(seconds_to_time_str(t))

  if chart_info_dict['singles or doubles'] == 'Singles':
    stance = _stances.Stances(style='singles')
  if chart_info_dict['singles or doubles'] == 'Doubles':
    stance = _stances.Stances(style='doubles')
  _, all_holds = plot_chart.js_arrows(line_df, stance)

  # Construct timelines. Ordered: hold, twist, then 1-2-3 custom
  base_timelines = [
    e_struct_timelines.binary_timeline(line_df, 'Hold', 'Hold'),
    e_struct_timelines.twist(line_df),
  ]
  add_cols = compare_d['timeline tags']
  add_names = [chart_compare.rename_tag(t) for t in add_cols]
  print(add_names)
  add_timelines = [
    e_struct_timelines.binary_timeline(line_df, col, name)
    for col, name in zip(add_cols, add_names)
  ]

  chart_card_dict = {
    # 'groups': groups,
    'xticks': xticks,
    'xlabels': xlabels,
    'nps': e_struct_timelines.nps(line_df),
    # 'hold_timeline': e_struct_timelines.hold_timeline(all_holds),
    'timelines': base_timelines + add_timelines,
  }
  return convert_dict_to_js_lists(chart_card_dict)

# 3
def chart_details(nm, line_df, groups, chart_info_dict):
  if chart_info_dict['singles or doubles'] == 'Singles':
    num_panels = 5
    stance = _stances.Stances(style='singles')
  if chart_info_dict['singles or doubles'] == 'Doubles':
    num_panels = 10
    stance = _stances.Stances(style='doubles')

  # [preview, 1, 2, 3, ...]
  chart_details_struct = []
  if len(groups) > 1:
    preview_start = groups[1][0]
  else:
    preview_start = groups[0][0]
  preview_end = preview_start + 16
  groups.insert(0, (preview_start, preview_end))
  section_names = []
  names = ['preview'] + [x+1 for x in range(len(groups))]

  _, all_holds = plot_chart.js_arrows(line_df, stance)

  _, all_holds = plot_chart.js_arrows(line_df, stance)

  for name, group in zip(names, groups):
    dfs = line_df.iloc[group[0]:group[1]]
 
    arrows, holds = plot_chart.js_arrows(dfs, stance)
    annot_times, annotations = plot_chart.js_line_annotations(dfs)

    stats = plot_chart.get_section_stats(group, line_df)
    min_time = min(dfs['Time'])
    max_time = max(dfs['Time'])
    dt = stats['Median time since downpress']
    times = [float(t) for t in np.arange(min_time, max_time + dt, dt)]
    long_holds = plot_chart.get_long_holds(min_time, max_time, all_holds)

    time_labels = [f'{t:.2f}' if i % 4 == 0 else '' for i, t in enumerate(times)]

    long_holds = plot_chart.get_long_holds(min_time, max_time, all_holds)

    chart_details_dict = {
      'section_name': name,
      'num_panels': num_panels,
      'num_lines': len(times),
      'times': times,
      'holds': holds + long_holds,
      'plot_start_time': float(max_time + dt/2),
      'plot_end_time': float(min_time - dt/2),
      'arrows': arrows,
      'holds': holds + long_holds,
      'annots': [annot_times, annotations],
    }
    chart_details_struct.append(convert_dict_to_js_lists(chart_details_dict))

    section_names.append(f'{name}, {seconds_to_time_str(min_time)}-{seconds_to_time_str(max_time)}')

  # Update chart_info_dict, accessible in HTML/Jinja
  new_info = {
    'num_panels': num_panels,
    'num_chart_sections': len(groups) - 1,  # -1 for preview
    'section_names': section_names,
  }
  return chart_details_struct, new_info


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


def seconds_to_time_str(seconds):
  mins = round(seconds) // 60
  sec = round(seconds) % 60
  return f'{mins}:{sec:02d}'

'''
  Run
'''
def run_single(nm):
  line_df = pd.read_csv(inp_dir_d + f'{nm}.csv', index_col=0)
  all_line_dfs, groups, comb_to_indiv = hmm_segment.segment(line_df)

  chart_card_struct = chart_card(nm, line_df, groups, compare_d, chart_info_dict)
  compare_d = chart_compare.run_single(nm)

  chart_info_dict = chart_info(nm, compare_d)
  chart_card_struct = chart_card(nm, line_df, groups, compare_d)
  chart_details_struct, new_info = chart_details(nm, line_df, groups, chart_info_dict)

  # Info parsed into python dict, usable in HTML, not just javascript
  chart_info_dict.update(new_info)

  struct = [
    convert_dict_to_js_lists(chart_info_dict),
    chart_card_struct,
    chart_details_struct,
  ]

  with open(out_dir + f'{nm}.pkl', 'wb') as f:
    pickle.dump(struct, f)

  if os.environ.get('LOCAL_FRAMEWORK') == 'True':
    print('Detected local laptop env: Saving to piu-app ...')
    app_dir = f'../../piu-app/data/'
    with open(app_dir + f'{nm}.pkl', 'wb') as f:
      pickle.dump(struct, f)
  return


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
  # nm = 'Love is a Danger Zone - BanYa S7 arcade'
  nm = 'Imagination - SHK S17 arcade'
  # nm = 'Tepris - Doin S17 arcade'
  # nm = '8 6 - DASU S20 arcade'
  # nm = 'The End of the World ft. Skizzo - MonstDeath S20 arcade'
  # nm = 'Bad Apple!! feat. Nomico - Masayoshi Minoshima S17 arcade'
  # nm = 'Ugly Dee - Banya Production D15 arcade'

  # Doubles
  # nm = 'Mitotsudaira - ETIA. D19 arcade'
  # nm = 'Rock the house - Matduke D22 arcade'
  # nm = 'Ugly Dee - Banya Production D15 arcade'

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