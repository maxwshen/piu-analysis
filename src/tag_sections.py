'''
    Tag charts
'''
import _config, util
import sys, os, pickle, fnmatch, datetime, subprocess, functools, re
import numpy as np, pandas as pd
from collections import defaultdict, Counter

import d_annotate
import plot_chart
import _qsub, _stepcharts, hmm_segment, chart_compare

import warnings
warnings.filterwarnings('ignore')

# Default params
inp_dir_a = _config.OUT_PLACE + 'a_format_data/'
inp_dir_d = _config.OUT_PLACE + 'd_annotate/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

scinfo = _stepcharts.SCInfo()



def edit_section_features(ft_ds):
  # Edit features. E.g., only want 'Mid4' label if 100% of section is Mid4
  # We featurize sections using the same process as featurizing entire charts, where recognizing percent differences in mid4 is desirable
  cols_100_or_none = [
    'Mid4 doubles - frequency',
    'Mid6 doubles - frequency',
    'Side3 singles - frequency',
    'Twist angle - none - frequency',
  ]
  for ftd in ft_ds:
    for col in cols_100_or_none:
      if ftd[col] < 1:
        ftd[col] = 0
  return ft_ds


'''
  Descriptions
'''
def get_section_stats(dfs):
  dp_dfs = dfs[dfs['Has downpress']]
  
  bs = dp_dfs['Beat since'][1:]
  bsd = dp_dfs['Beat since downpress'][1:]
  ts = dp_dfs['Time since'][1:]
  median_bpm = np.median([bpm for bpm in dfs['BPM'] if bpm < 999])
  stats = {
    'Start beat': min(dfs['Beat']),
    'End beat': max(dfs['Beat']),
    'Start time': min(dfs['Time']),
    'End time': max(dfs['Time']),
    'Time length': max(dfs['Time']) - min(dfs['Time']),
    'Median beat since': np.median(bs),
    'Median beat since downpress': np.median(bsd),
    'Median time since': np.median(ts),
    'Median bpm': median_bpm,
    'Beat time inc from bpm': 60 / (median_bpm),
    'Num. lines': len(dfs),
    'Num. downpress lines': len(dp_dfs),
  }
  
  stats['Median nps'] = (stats['Median bpm'] / stats['Median beat since downpress']) / 60
  return stats


def get_description(tags):
  '''
    Label section with a single "primary" high-level tag
    Can modify or remove adjectives here
  '''
  highlevel = ['Bracket jump run', 'Jump run', 'Run', 'Drill', 'Bracket', 'Hold run', 'Hold taps', 'Jump', 'Jack', 'Footswitch', 'Hold', 'None']
  replace = {
    'Bracket jump run': 'Bracket jump',
    'Jump run': 'Jump',
  }
  extra = ['Hands', 'Stairs, singles', 'Stairs, doubles', 'Broken stairs, doubles',
    'Bracket', 'Splits', 'Staggered hit',  'Side3 singles', 'Mid4 doubles', 'Mid6 doubles', 'Spin', 'Double step', 'Bracket footswitch', 'Hold footswitch', 'Hold footslide'
  ]

  ordered_tags = list(tags.keys())
  def idx_of(tag):
    if tag == 'None':
      return 1000
    return ordered_tags.index(tag) if tag in ordered_tags else np.inf
  primary_annot = min(highlevel, key=idx_of)
  if primary_annot == 'None':
    return '', '', '', []

  # Prefer hold taps, hold run over hold if possible
  if primary_annot == 'Hold':
    for prec in ['Hold taps', 'Hold run']:
      if prec in tags:
        primary_annot = prec
  if primary_annot == 'Run':
    for prec in ['Drill']:
      if prec in tags:
        primary_annot = prec

  adjs = tags[primary_annot] if primary_annot in tags else []
  adjs = adjs + [tag for tag in extra if tag in tags]
  adjs = reduce_adjs(adjs)
  adjs = remove_adjs(adjs)
  adjs = relabel_adjs(adjs)

  bpm_annot = [a for a in adjs if 'bpm' in a][0]
  bpm_annot = '' if 'nan' in bpm_annot else bpm_annot
  nps_annot = [a for a in adjs if 'nps' in a][0]
  nps_annot = '' if 'nan' in nps_annot else nps_annot
  adjs = [a for a in adjs if 'nps' not in a and 'bpm' not in a]
  adjs = [a.capitalize() for a in adjs]

  primary_annot = replace.get(primary_annot, primary_annot)
  return primary_annot, bpm_annot, nps_annot, adjs


def reduce_adjs(adjs):
  # Collapse redundant adjectives: e.g., keep mid4 and remove mid6
  groups = [
    ['Mid4 doubles', 'Mid6 doubles'],
    ['Stairs, doubles', 'Stairs, singles'],
  ]
  for group in groups:
    if sum(x in adjs for x in group) > 1:
      adjs.remove(group[-1])
  return adjs


def remove_adjs(adjs):
  remove = ['small movements', 'front-facing']
  return [a for a in adjs if a not in remove]


def relabel_adjs(adjs):
  relabel = {
      'Stairs, singles': '5-panel stairs',
      'Stairs, doubles': '10-panel stairs', 
      'Broken stairs, doubles': 'Broken 10-panel stairs',
      'Staggered hit': 'Rolling hits',
      'Side3 singles': 'Side3',
      'Mid4 doubles': 'Mid4',
      'Mid6 doubles': 'Mid6',
    }
  return [relabel.get(adj, adj) for adj in adjs]


'''
  Plotting
'''
def format_time(secs):
    ms = secs // 60
    ss = int(secs % 60)
    return f'{ms:.0f}:{ss:02d}'


def ensure_min_rect_heights(min_height, section_data):
  ys, heights = [], []
  for sec_num, section in enumerate(section_data):
    ds = section_data[section]
    start, end = ds['Start time'], ds['End time']
    time_len = max(min_height, end - start)
    ys.append(sum(heights))
    heights.append(time_len)
  return ys, heights


def plot_summary(nm, section_data, line_df, groups, comb_to_indiv, out_fn):
  '''
    section_data
    summary_annots
    bpm_annots
    adj_annots
  '''
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import seaborn as sns

  # summary_width = 1.2
  summary_width = 10
  RECT_WIDTH = 2
  MIN_RECT_HEIGHT = 3

  norm = mpl.colors.Normalize(vmin=-15, vmax=15)
  cmap = mpl.cm.nipy_spectral
  float_to_color = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
  color_of = lambda x: float_to_color.to_rgba(x)

  ys, heights = ensure_min_rect_heights(MIN_RECT_HEIGHT, section_data)
  expanded_time_len = sum(heights)
  summary_height = expanded_time_len * 0.06

  fig, axes = plt.subplots(1, 5,
                            figsize=(summary_width, summary_height),
                            gridspec_kw={'width_ratios': [1, 1, 0.1, 1, 2]},
                            sharey=True)
  [group_ax, main_ax, nps_ax, bpm_ax, annot_ax] = axes

  start_time, end_time = ys[0], ys[-1] + heights[-1]
  main_ax.set_ylim([start_time-0.1, end_time+0.1])
  main_ax.invert_yaxis()

  # Plot rectangles and adjectives
  yticklocs = []
  interval_times = []
  for sec_num, section in enumerate(section_data):
    ds = section_data[section]
    nps = float(ds['NPS annot'].replace(' nps', ''))
    color = color_of(nps)
    start_time, end_time = ds['Start time'], ds['End time']
    
    y_start = ys[sec_num]
    y_end = ys[sec_num] + heights[sec_num]

    rect = mpl.patches.Rectangle(
      (0 - RECT_WIDTH/2, y_start), 
      RECT_WIDTH, y_end,
      linewidth=0, facecolor=color)
    main_ax.add_patch(rect)
    
    if sec_num == 0:
      yticklocs += [y_start, y_end]
      interval_times += [ds['Start time'], ds['End time']]
    else:
      yticklocs += [y_end]
      interval_times += [ds['End time']]
        
    primary_annot = ds['Primary annot']
    text_y_offset = 0.75
    y_annot = np.mean([y_start, y_end]) + text_y_offset
    main_ax.text(0, y_annot, primary_annot, ha='center')

    nps_ax.text(0, y_annot, ds['NPS annot'].replace(' nps', ''), ha='left')
    bpm_ax.text(0, y_annot, ds['BPM annot'].replace(' at ', ', '), ha='left')
    annot_ax.text(0, y_annot, ds['Adjectives annot'], ha='left')

  # Plot groups (for chart details, less than 10)
  palette = sns.color_palette()
  for i, group in enumerate(groups):
    indivs = comb_to_indiv[i]
    start_i, end_i = indivs[0], indivs[-1]
    y_start = ys[start_i]
    y_end = ys[end_i] + heights[end_i]
    
    section_rect_width = 0.5
    rect = mpl.patches.Rectangle(
      (0 - section_rect_width/2, y_start), 
      section_rect_width, y_end,
      linewidth=0, color=palette[i])
    group_ax.add_patch(rect)
    group_ax.plot((-1, 1), (y_start, y_start),
                  color='black', linewidth=1, linestyle='-')
    group_ax.plot((-1, 1), (y_end, y_end),
                  color='black', linewidth=1, linestyle='-')
    group_ax.text(-1, np.mean([y_start, y_end]), f'{i+1}', ha='right', va='center')


  # Formatting
  # y ticks
  interval_times = sorted(list(set([round(t, 1) for t in interval_times])))  
  main_ax.set_yticks(yticklocs)
  main_ax.set_yticklabels([format_time(t) for t in interval_times])
  main_ax.set_ylabel('Time')
  main_ax.yaxis.set_tick_params(labelleft=True)
  main_ax.tick_params('x', width=0)
  main_ax.set_xticklabels([])
  main_ax.set_xlim([-1, 1])
  # right_ax.set_ylim([min(yticklocs)-0.1, max(yticklocs)+0.1])
  main_ax.grid(axis='y')
  
  nps_ax.set_title('NPS');
  nps_ax.set_xlim([0, 5])
  nps_ax.axis('off')

  group_ax.axis('off')
  group_ax.set_xlim([-5, 1])
  group_ax.set_title('Section')

  bpm_ax.set_xlim([0, 3])
  bpm_ax.axis('off')

  annot_ax.axis('off')

  sns.despine(bottom=True, left=True)
  main_ax.set_title(f'{nm}\n')

  fig.tight_layout()
  fig.patch.set_facecolor('white')
  fig.savefig(out_fn, bbox_inches='tight')
  plt.close()
  return


'''
  Primary
'''
def summary(nm, all_line_dfs, line_df, groups, comb_to_indiv, out_fn):
  # Tag section, comparing subchart to all charts in context (level-3 to level)
  section_stats = [get_section_stats(dfs) for dfs in all_line_dfs]
  ft_ds = [d_annotate.featurize(dfs) for dfs in all_line_dfs]
  ft_ds = edit_section_features(ft_ds)
  df_fts = [pd.DataFrame(fts, index=[nm]) for fts in ft_ds]

  # Annotate
  context_df = chart_compare.get_context_df(nm)
  context_df = context_df[~(context_df['Name (unique)'] == nm)]

  section_data = {}
  aug_contexts = []
  for i, df_ft in enumerate(df_fts):
    df_ft['Name (unique)'] = nm
    aug_context = df_ft.append(context_df)
    aug_contexts.append(aug_context)
    tags = chart_compare.tag_chart(aug_context, nm)
    section_data[i] = {
      'Tags': tags,
    }
    section_data[i].update(section_stats[i])

    primary_annot, bpm_annot, nps_annot, adjs = get_description(tags)
    section_data[i]['Primary annot'] = primary_annot
    section_data[i]['BPM annot'] = bpm_annot
    section_data[i]['NPS annot'] = nps_annot
    section_data[i]['Adjectives annot'] = ', '.join(adjs)

  plot_summary(nm, section_data, line_df, groups, comb_to_indiv, out_fn)
  return


'''
  Run
'''
def run_single(nm):
  '''
    HMM segment a chart by 'time since downpress'
    Tag section, comparing subchart to all charts in context (level-3 to level)
  '''
  line_df = pd.read_csv(inp_dir_d + f'{nm}.csv', index_col=0)

  all_line_dfs, groups, comb_to_indiv = hmm_segment.segment(line_df)
  print(f'Found groups:', groups)

  stepchart_out_dir = out_dir + f'{nm}/'
  util.ensure_dir_exists(stepchart_out_dir)

  print('Plotting stepchart card ...')
  summary_fn = stepchart_out_dir + f'{nm} card.png'
  summary(nm, all_line_dfs, line_df, groups, comb_to_indiv, summary_fn)

  # Plot chart by section
  print('Plotting sections ...')
  sord = scinfo.name_to_singleordouble(nm)
  artist = plot_chart.Artist(sord)
  total_lines_done = 0
  for i, group in enumerate(groups):
    sec_num = i+1
    out_fn = stepchart_out_dir + f'{nm} {sec_num}.png'
    artist.plot_section(line_df, group, out_fn)

    total_lines_done += group[1] - group[0]
    print(f'Group {i+1} done: {total_lines_done}/{len(line_df)}; {total_lines_done/len(line_df):.1%}')

  return


@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  nm = 'Super Fantasy - SHK S16 arcade'
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
  # nm = 'Nakakapagpabagabag - Dasu feat. Kagamine Len S18 arcade'

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