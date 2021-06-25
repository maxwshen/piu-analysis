'''
  Annotate charts and movements.
  Used for
  - Chart clustering
  - Predicting difficulty
  - Tagging charts
'''
import _config, util
import sys, os, pickle, fnmatch, datetime, subprocess, functools, re
import numpy as np, pandas as pd
from collections import defaultdict, Counter

import b_graph, segment, _qsub
import _notelines, _movement

# Default params
inp_dir_b = _config.OUT_PLACE + 'b_graph/'
inp_dir_segment = _config.OUT_PLACE + 'segment/'
inp_dir_c = _config.OUT_PLACE + 'c_dijkstra/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

mover = None

import _annotate_local, _annotate_global
local_funcs = _annotate_local.funcs
global_funcs = _annotate_global.funcs

annots = list(local_funcs.keys()) + list(global_funcs.keys())
annot_types = _annotate_local.annot_types
annot_types.update(_annotate_global.annot_types)

add_annots = {
  'Notes per second since downpress': float,
}
annots += list(add_annots.keys())
annot_types.update(add_annots)


'''
  Parsing
'''
def get_ds(row1, row2):
  '''
    d keys = ['limb_to_pos', 'limb_to_heel_action', 'limb_to_toe_action']
    d[key][limb] = pos / heel action / toe action
  '''
  if row1 is None:
    return None, parse_sa(row2['Stance action'])
  sa1, sa2 = row1['Stance action'], row2['Stance action']
  d1, d2 = parse_sa(sa1), parse_sa(sa2)
  return d1, d2


@functools.lru_cache(maxsize=None)
def parse_sa(sa):
  return mover.parse_stanceaction(sa)


'''
  Primary
'''
def annotate_general(df):
  td = np.array(df['Time'].iloc[1:]) - np.array(df['Time'].iloc[:-1])
  df['Time since'] = [0] + list(td)

  tsd = [0]
  buffer = 0
  for i in range(1, len(df)):
    row = df.iloc[i]
    prev_line = df['Line with active holds'].iloc[i-1]
    line = row['Line with active holds']
    buffer += row['Time since']
    if _notelines.has_downpress(line) and prev_line.replace('1', '2') != line:
      tsd.append(buffer)
      buffer = 0
    else:
      tsd.append(buffer)
  df['Time since downpress'] = tsd

  df['Has downpress'] = [_notelines.has_downpress(line) for line in df['Line']]

  nps = [np.nan]
  has_dp_adj = [True]
  for i in range(1, len(df)):
    prev_line = df['Line with active holds'].iloc[i-1]
    line = df['Line with active holds'].iloc[i]
    if df['Has downpress'].iloc[i] and prev_line.replace('1', '2') != line:
      nps.append(1 / df['Time since downpress'].iloc[i])
      has_dp_adj.append(True)
    else:
      nps.append(np.nan)
      has_dp_adj.append(False)
  df['Notes per second since downpress'] = nps
  df['Has downpress adj.'] = has_dp_adj

  body_angles = []
  for i, row in df.iterrows():
    _, d = get_ds(None, row)
    lpos = mover.pos_to_center[d['limb_to_pos']['Left foot']]
    rpos = mover.pos_to_center[d['limb_to_pos']['Right foot']]
    body_angles.append(body_angle_from_pos(lpos, rpos))
  df['Body angle'] = body_angles

  return df


def annotate_local(df):
  # Annotate using previous and current rows only
  cdd = defaultdict(list)
  for i in range(len(df)):
    prev_row = None if i == 0 else df.iloc[i-1]
    row = df.iloc[i]

    for name, func in local_funcs.items():
      res = func(prev_row, row)
      cdd[name].append(res)

  for col in cdd:
    df[col] = cdd[col]
  return df


def annotate_global(df):
  cdd = {}
  for name, func in global_funcs.items():
    res = func(df)
    cdd[name] = res
  for col in cdd:
    df[col] = cdd[col]
  return df


'''
  Annotate general
'''
def body_angle_from_pos(lpos, rpos, verbose = False):
  '''
    Left foot to right foot vector. Exactly to the right = 0 degrees.
    Degrees increment counterclockwise.
  '''
  xdiff = rpos[0] - lpos[0]
  ydiff = rpos[1] - lpos[1]
  if xdiff == 0 and ydiff == 0:
    angle = 0
  else:
    angle = np.arccos(xdiff / np.linalg.norm([xdiff, ydiff]))
    angle = np.degrees(angle)
  if ydiff > 0:
    angle = 360 - angle
  return angle


'''
  Featurize
'''
def featurize(df):
  '''
    Parse bools by fraction of downpress lines
    - Include stats on nps for these subsets
    Special parse for twist angle
    Parse floats by percentiles among downpress lines

    Use 'Has downpress adj.', which counts 1->2 as 1 downpress, not 2. This aligns the nm. downpress lines with user expectations
  '''
  one_hot_encode(df, 'Twist angle', ['90', 'diagonal', '180'])

  dfs = df[df['Has downpress adj.']]

  type_to_featurizer = {
    bool: bool_featurize,
    float: float_featurize,
  }
  all_stats = {}
  for annot in annots:
    f = type_to_featurizer[annot_types[annot]]
    stats = f(dfs, annot)
    all_stats.update(stats)
  return all_stats


def bool_featurize(df, col):
  nps = np.array(df[df[col]]['Notes per second since downpress'])
  if len(nps) == 0:
    nps = np.array([np.nan])
  stats = {
    f'{col} - frequency': sum(df[col]) / len(df),
    f'{col} - 50% nps':   nan_to_zero(np.nanmedian(nps)),
    f'{col} - 80% nps':   nan_to_zero(np.nanpercentile(nps, 80)), 
    f'{col} - 99% nps':   nan_to_zero(np.nanpercentile(nps, 99)), 
  }
  return stats


def float_featurize(df, col):
  # ex: Travel (mm)
  data = np.array(df[col])
  stats = {
    f'{col} - 50%': nan_to_zero(np.nanmedian(data)),
    f'{col} - 80%': nan_to_zero(np.nanpercentile(data, 80)), 
    f'{col} - 99%': nan_to_zero(np.nanpercentile(data, 99)), 
  }
  return stats


def nan_to_zero(x):
  return 0 if np.isnan(x) else x


def one_hot_encode(df, ft, cats):
  cols = []
  for cat in cats:
    col = f'{ft} - {cat}'
    df[col] = (df[ft] == cat)
    cols.append(col)
  
  global annots
  global annot_types
  # Only edit 1st time, when running multiple times with qsub
  if ft in annots:
    annots.remove(ft)
    annots += cols

    del annot_types[ft]
    for col in cols:
      annot_types[col] = bool
  return


'''
  Run
'''
def run_single(nm):
  move_skillset = 'basic'
  print(nm, move_skillset)

  line_nodes, line_edges_out, line_edges_in = b_graph.load_data(inp_dir_b, nm)

  steptype = line_nodes['init']['Steptype']
  global mover
  mover = _movement.Movement(style=steptype, move_skillset=move_skillset)

  _annotate_local.mover = mover
  _annotate_global.mover = mover
  _annotate_local.get_ds = get_ds
  _annotate_global.get_ds = get_ds

  df = pd.read_csv(inp_dir_c + f'{nm} {move_skillset}.csv', index_col=0)
  df = annotate_general(df)
  df = annotate_local(df)
  df = annotate_global(df)
  df.to_csv(out_dir + f'{nm} {move_skillset}.csv')

  # Featurize for chart tagging, clustering, and predicting difficulty
  fts = featurize(df)
  df_fts = pd.DataFrame(fts, index=[nm])
  df_fts.to_csv(out_dir + f'{nm}_{move_skillset}_features.csv')
  return


@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  # nm = 'Super Fantasy - SHK S19 arcade'
  nm = 'Uranium - Memme S19 arcade'
  # nm = 'Last Rebirth - SHK S15 arcade'
  # nm = 'Tepris - Doin S17 arcade'
  # nm = 'Super Fantasy - SHK S7 arcade'
  # nm = 'Super Fantasy - SHK S4 arcade'
  # nm = 'Final Audition 2 - BanYa S7 arcade'
  # nm = 'Sorceress Elise - YAHPP S23 arcade'
  # nm = 'Super Fantasy - SHK S10 arcade'
  # nm = '1950 - SLAM S23 arcade'
  # nm = 'HTTP - Quree S21 arcade'
  # nm = '8 6 - DASU S20 arcade'
  # nm = 'Shub Sothoth - Nato & EXC S25 remix'
  # nm = 'The End of the World ft. Skizzo - MonstDeath S20 arcade'
  # nm = 'Loki - Lotze S21 arcade'
  # nm = 'Native - SHK S20 arcade'
  # nm = 'PARADOXX - NATO & SLAM S26 remix'
  # nm = 'BEMERA - YAHPP S24 remix'
  # nm = 'HEART RABBIT COASTER - nato S23 arcade'
  # nm = 'F(R)IEND - D_AAN S23 arcade'
  # nm = 'Pump me Amadeus - BanYa S11 arcade'
  # nm = 'King of Sales - Norazo S21 arcade'
  # nm = 'Hyperion - M2U S20 shortcut'
  # nm = 'Final Audition Ep. 2-2 - YAHPP S22 arcade'
  # nm = 'Achluoias - D_AAN S24 arcade'
  # nm = 'Awakening - typeMARS S16 arcade'

  # Doubles
  # nm = 'Mitotsudaira - ETIA. D19 arcade'
  # nm = 'Trashy Innocence - Last Note. D16 arcade'
  # nm = '8 6 - DASU D21 arcade'
  # nm = 'Bad End Night - HitoshizukuP x yama D18 arcade'

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
    elif sys.argv[1] == 'run_single':
      run_single(sys.argv[2])