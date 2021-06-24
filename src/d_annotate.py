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

# min. lines in a globally annotated section
GLOBAL_MIN_LINES_SHORT = 4
GLOBAL_MIN_LINES_LONG = 8

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
    line = row['Line with active holds']
    buffer += row['Time since']
    if _notelines.has_downpress(line):
      tsd.append(buffer)
      buffer = 0
    else:
      tsd.append(buffer)
  df['Time since downpress'] = tsd

  df['Has downpress'] = [_notelines.has_downpress(line) for line in df['Line']]

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
  local_funcs = {
    # Line only
    'Hold': is_hold,
    'Hold taps': is_hold_taps,
    'Splits': is_splits,
    # Movement required
    'Jump': is_jump,
    'Bracket': is_bracket,
    'Double step': is_doublestep,
    'Twist angle': twist_angle,
    'Footswitch': is_footswitch,
    'Jack': is_jack,
    'Bracket footswitch': is_bracket_footswitch,
    'Hold tap single foot': is_hold_tap,
    'Hold footslide': is_hold_footslide,
    'Hold footswitch': is_hold_footswitch,
    'Staggered hit': is_multiline,
    'Hands': is_hands,
    'Travel (mm)': travel,
  }

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
  global_funcs = {
    # Line only
    'Run': run,
    'Hold run': hold_run,
    'Drill': drill,
    'Bracket drill': bracket_drill,
    'Irregular rhythm': irregular_rhythm,
    'Bracket jump run': bracket_jump_run,
    'Side3 singles': side3_singles,
    'Mid4 doubles': mid4_doubles,
    'Mid6 doubles': mid6_doubles,
    # Movement required
    'Run with brackets': bracket_run,
    'Jump run': jump_run,
    'Stairs, singles': singles_stair,
    'Stairs, doubles': doubles_stair,
    'Broken stairs, doubles': doubles_broken_stair,
    'Spin': spin,
  }
  cdd = {}
  for name, func in global_funcs.items():
    res = func(df)
    cdd[name] = res
  for col in cdd:
    df[col] = cdd[col]
  return df


'''
  Local - line only
'''
def is_hold(row1, row2):
  line2 = row2['Line with active holds']
  return any(x in line2 for x in list('24'))


def is_hold_taps(row1, row2):
  line2 = row2['Line with active holds']
  active_hold = '4' in line2
  has_tap = any(x in line2 for x in list('12'))
  return active_hold and has_tap


def is_splits(row1, row2):
  line2 = row2['Line with active holds']
  if len(line2) == 10:
    if any(x in line2[:2] for x in list('12')):
      if any(x in line2[-2:] for x in list('12')):
        return True
  return False

'''
  Local - line + movement
  x Twist (90, diagonal, 180)
  x	Jump
  x	Double step
  x	Jacks
  x	Footswitches
  x	Bracket
  x Bracket footswitch (Rock the House D22)
  x	Hold taps (86 d21)
  x	Hold footchange (Iolite Sky D24)
  x Hold footslide (86 full d23)
  x	Multiline hits / staggered hits / staggered brackets
  x Hands
  x	Bracket jump
  x	Travel
    - Score can depend on time between lines (handle later)
    - Later, tabulate like bpm: 99 percentile movement, and 80 pct.
'''
def is_jump(row1, row2):
  if row1 is None:
    return False
  return row2['jump'] > 0


def is_doublestep(row1, row2):
  '''
    Different logic than _movement.double_step_cost
    - Count taps in active holds
    - Reused limb must be the only limb with downpresses
    - Forgive 1/3 -> 2 with same limb on same pad
  '''
  if row1 is None:
    return False
  d1, d2 = get_ds(row1, row2)
  limbs = list(limb for limb in d2['limb_to_pos']
                    if limb in d1['limb_to_heel_action'])

  if row1['jump'] or row2['jump']:
    return False

  doublestep_prev = set(list('123'))
  doublestep_curr = set(list('12'))
  for limb in limbs:
    prev_heel = d1['limb_to_heel_action'][limb] in doublestep_prev
    prev_toe = d1['limb_to_toe_action'][limb] in doublestep_prev
    curr_heel = d2['limb_to_heel_action'][limb] in doublestep_curr
    curr_toe = d2['limb_to_toe_action'][limb] in doublestep_curr

    prev_step = prev_heel or prev_toe
    curr_step = curr_heel or curr_toe

    limb_cost = 0
    if prev_step and curr_step:
      limb_cost = 1

    # Forgive 1/3->2 with same limb on same pad
    prev_heel_a = d1['limb_to_heel_action'][limb]
    curr_heel_a = d2['limb_to_heel_action'][limb]
    prev_toe_a = d1['limb_to_toe_action'][limb]
    curr_toe_a = d2['limb_to_toe_action'][limb]
    heel_pad_match = d1['limb_to_pos'][limb] == d2['limb_to_pos'][limb]
    toe_pad_match = d1['limb_to_pos'][limb] == d2['limb_to_pos'][limb]
    if prev_heel_a in list('13') and curr_heel_a == '2' and heel_pad_match:
      limb_cost = 0
    elif prev_toe_a in list('13') and curr_toe_a == '2' and toe_pad_match:
      limb_cost = 0

    if limb_cost > 0:
      return True
  return False


def twist_angle(row1, row2) -> str:
  # ['none', '90', 'diagonal', '180']
  d1, d2 = get_ds(row1, row2)
  body_angle = row2['Body angle']
  angle = min(body_angle, 360 - body_angle)
  leniency = 15
  if angle < 90 - leniency:
    return 'none'
  elif 90 - leniency <= angle <= 90 + leniency:
    return '90'
  elif 90 + leniency < angle <= 180 - leniency:
    return 'diagonal'
  else:
    return '180'


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



def is_footswitch(row1, row2):
  # If exists a pad that was hit by 1 limb, then other limb
  # More general than movement annotations to capture Native S20
  d1, d2 = get_ds(row1, row2)
  if row1 is None:
    return False
  if row2['jump']:
    return False

  old_limbs = ['Left foot', 'Right foot']
  new_limbs = ['Right foot', 'Left foot']
  for old_limb, new_limb in zip(old_limbs, new_limbs):
    prev_panels = []
    if any(x in d1['limb_to_heel_action'][old_limb] for x in list('13')):
      panel = mover.pos_to_heel_panel[d1['limb_to_pos'][old_limb]]
      prev_panels.append(panel)
    if any(x in d1['limb_to_toe_action'][old_limb] for x in list('13')):
      panel = mover.pos_to_toe_panel[d1['limb_to_pos'][old_limb]]
      prev_panels.append(panel)

    if any(x in d2['limb_to_heel_action'][new_limb] for x in list('12')):
      panel = mover.pos_to_heel_panel[d2['limb_to_pos'][new_limb]]
      if panel in prev_panels:
        return True
    if any(x in d2['limb_to_toe_action'][new_limb] for x in list('12')):
      panel = mover.pos_to_toe_panel[d2['limb_to_pos'][new_limb]]
      if panel in prev_panels:
        return True
  return False


def is_jack(row1, row2):
  # If exists a pad that was hit by 1 limb twice
  d1, d2 = get_ds(row1, row2)
  if row1 is None:
    return False
  if row2['jump']:
    return False

  res = False
  old_limbs = ['Left foot', 'Right foot']
  new_limbs = ['Left foot', 'Right foot']
  for old_limb, new_limb in zip(old_limbs, new_limbs):
    prev_panels = []
    if any(x in d1['limb_to_heel_action'][old_limb] for x in list('13')):
      panel = mover.pos_to_heel_panel[d1['limb_to_pos'][old_limb]]
      prev_panels.append(panel)
    if any(x in d1['limb_to_toe_action'][old_limb] for x in list('13')):
      panel = mover.pos_to_toe_panel[d1['limb_to_pos'][old_limb]]
      prev_panels.append(panel)

    if any(x in d2['limb_to_heel_action'][new_limb] for x in list('12')):
      panel = mover.pos_to_heel_panel[d2['limb_to_pos'][new_limb]]
      if panel in prev_panels:
        res = True
    if any(x in d2['limb_to_toe_action'][new_limb] for x in list('12')):
      panel = mover.pos_to_toe_panel[d2['limb_to_pos'][new_limb]]
      if panel in prev_panels:
        res = True
  return res


def is_bracket_footswitch(row1, row2):
  # If exists a pad that was hit by 1 limb, then other limb
  d1, d2 = get_ds(row1, row2)
  if row1 is None:
    return False

  old_limbs = ['Left foot', 'Right foot']
  new_limbs = ['Right foot', 'Left foot']
  for old_limb, new_limb in zip(old_limbs, new_limbs):
    old_pos = d1['limb_to_pos'][old_limb]
    new_pos = d1['limb_to_pos'][new_limb]
    if old_pos == new_pos and new_pos in mover.bracket_pos:
      prev_ha = d1['limb_to_heel_action'][old_limb]
      prev_ta = d1['limb_to_toe_action'][old_limb]
      curr_ha = d1['limb_to_heel_action'][new_limb]
      curr_ta = d1['limb_to_toe_action'][new_limb]

      if set([prev_ha, prev_ta, curr_ha, curr_ta]) == set([1]):
        return True

  return False


def is_bracket(row1, row2):
  # Any foot in bracket position and downpressing
  d1, d2 = get_ds(row1, row2)
  for limb in ['Left foot', 'Right foot']:
    pos = d2['limb_to_pos'][limb]
    if pos in mover.bracket_pos:
      if any(x in d2['limb_to_heel_action'][limb] for x in list('124')):
        if any(x in d2['limb_to_toe_action'][limb] for x in list('124')):
          return True
  return False


def is_bracket_jump(row1, row2):
  if row1 is None:
    return False
  d1, d2 = get_ds(row1, row2)
  rpos = d2['limb_to_pos']['Right foot']
  lpos = d2['limb_to_pos']['Left foot']
  rmove = rpos != d1['limb_to_pos']['Right foot']
  lmove = lpos != d1['limb_to_pos']['Left foot']
  bracket = rpos in mover.bracket_pos or lpos in mover.bracket_pos
  return rmove and lmove and bracket


def is_hold_tap(row1, row2):
  # Hold-tap with single foot
  d1, d2 = get_ds(row1, row2)
  for limb in ['Left foot', 'Right foot']:
    if d2['limb_to_heel_action'][limb] == '4':
      if any(x in d2['limb_to_toe_action'][limb] for x in list('12')):
        return True
    if d2['limb_to_toe_action'][limb] == '4':
      if any(x in d2['limb_to_heel_action'][limb] for x in list('12')):
        return True
  return False


def is_hold_footslide(row1, row2):
  if row1 is None:
    return False
  return row2['hold_footslide'] > 0


def is_hold_footswitch(row1, row2):
  if row1 is None:
    return False
  return row2['hold_footswitch'] > 0


def is_multiline(row1, row2):
  if row1 is None:
    return False
  return 'multi' in row2['Node']


def is_hands(row1, row2):
  d1, d2 = get_ds(row1, row2)
  num_limbs = len(d2['limb_to_pos'].keys())
  return num_limbs == 4


def travel(row1, row2) -> float:
  # Average distance moved by heel and toe
  if row1 is None:
    return 0.0
  d1, d2 = get_ds(row1, row2)
  total_dist = 0
  for limb in ['Left foot', 'Right foot']:
    prev_heel = mover.pos_to_heel_coord[d1['limb_to_pos'][limb]]
    new_heel = mover.pos_to_heel_coord[d2['limb_to_pos'][limb]]
    prev_toe = mover.pos_to_toe_coord[d1['limb_to_pos'][limb]]
    new_toe = mover.pos_to_toe_coord[d2['limb_to_pos'][limb]]
    dist_heel = np.linalg.norm(prev_heel - new_heel, ord=2)
    dist_toe = np.linalg.norm(prev_toe - new_toe, ord=2)
    dist = np.mean(dist_heel + dist_toe)
    total_dist += dist
  return total_dist


'''
  Global - raw lines only. Needs larger context -- filter out <X in a row
  Cares about BPM
  x Runs
  x Hold runs
  x Drills
  x Bracket drill? (Loki S21, Sorceress Elise S23)
  x Irregular rhythm / gallops
  Don't care about BPM
  x Bracket jump run
  x Side 3 in singles
  x Middle 4 in doubles
  x Middle 6 in doubles
'''
def run(df):
  idxs = set()
  for i in range(1, len(df)):
    row1, row2 = df.iloc[i-1], df.iloc[i]
    if row2['Annotation'] == 'alternate' and '1' in row2['Line']:
      idxs.add(i-1)
      idxs.add(i)
  res = filter_short_runs(idxs, len(df), GLOBAL_MIN_LINES_LONG)  
  return res


def hold_run(df):
  idxs = set()
  for i in range(1, len(df)):
    row1, row2 = df.iloc[i-1], df.iloc[i]
    if row2['Annotation'] == 'alternate' and '2' in row2['Line']:
      idxs.add(i-1)
      idxs.add(i)
  res = filter_short_runs(idxs, len(df), 2*GLOBAL_MIN_LINES_LONG)  
  return res


def drill(df):
  idxs = set()
  MIN_DRILL_LEN = 5
  i, j = 0, 1
  while j < len(df):
    row1, row2 = df.iloc[i], df.iloc[j]
    is_1s = '1' in row1['Line'] and '1' in row2['Line']
    # alternate = row2['Annotation'] == 'alternate'
    if is_1s:
    # if is_1s and alternate:
      # valid start for drill
      k = j + 1
      while k < len(df):
        rowk = df.iloc[k]
        # Must repeat first two lines
        if (k - i) % 2 == 0:
          same_as = rowk['Line'] == row1['Line']
        else:
          same_as = rowk['Line'] == row2['Line']
        if same_as:
          k += 1
        else:
          break
      
      # Found drill
      if k - i >= MIN_DRILL_LEN:
        for idx in range(i, k):
          idxs.add(idx)
    i += 1
    j += 1

  res = filter_short_runs(idxs, len(df), MIN_DRILL_LEN)
  return res


def bracket_drill(df):
  idxs = set()
  MIN_DRILL_LEN = 4
  i, j = 0, 1
  while j < len(df):
    row1, row2 = df.iloc[i], df.iloc[j]
    is_bracket = row1['Line'].count('1') == 2 and row2['Line'].count('1') == 2
    if is_bracket:
      # valid start for drill
      k = j + 1
      while k < len(df):
        rowk = df.iloc[k]
        # Must repeat first two lines
        if (k - i) % 2 == 0:
          same_as = rowk['Line'] == row1['Line']
        else:
          same_as = rowk['Line'] == row2['Line']
        if is_bracket and same_as:
          k += 1
        else:
          break
      
      # Found drill
      if k - i >= MIN_DRILL_LEN:
        for idx in range(i, k):
          idxs.add(idx)
    i += 1
    j += 1

  res = filter_short_runs(idxs, len(df), MIN_DRILL_LEN)
  return res


def irregular_rhythm(df):
  # Time since is not a power of 2 of previous time since previous downpress
  # Epsilon needed to account for bpm changes
  epsilon = 0.05
  epsilon_close = lambda query, target: target-epsilon <= query <= target+epsilon
  res = [False]
  ts = list(df['Time since downpress'])
  has_dp = list(df['Has downpress'])
  ints = list(range(-10, 10))
  time_since_dp_at_prev_dp = ts[0]
  for i in range(1, len(ts)):
    if has_dp[i]:
      if time_since_dp_at_prev_dp == 0:
        ratio = 0
      else:
        ratio = np.log2(ts[i] / time_since_dp_at_prev_dp)
      regular = any(epsilon_close(ratio, i) for i in ints)
      time_since_dp_at_prev_dp = ts[i]
      res.append(not regular)
    else:
      res.append(False)
  return res


def bracket_jump_run(df):
  res = []
  lines = list(df['Line with active holds'])
  accept = lambda line: 3 <= line.count('1') <= 4
  idxs = [i for i, line in enumerate(lines) if accept(line)]
  res = filter_short_runs(idxs, len(lines), GLOBAL_MIN_LINES_SHORT)
  return res


def side3_singles(df):
  lines = list(df['Line with active holds'])
  if len(lines[0]) != 5:
    return [False]*len(df)

  left_accept = lambda line: line[-2:] == '00'
  left_idxs = [i for i, line in enumerate(lines) if left_accept(line)]
  left_res = filter_short_runs(left_idxs, len(lines), GLOBAL_MIN_LINES_LONG)

  right_accept = lambda line: line[:2] == '00'
  right_idxs = [i for i, line in enumerate(lines) if right_accept(line)]
  right_res = filter_short_runs(right_idxs, len(lines), GLOBAL_MIN_LINES_LONG)
  return [bool(l or r) for l, r in zip(left_res, right_res)]


def mid4_doubles(df):
  lines = list(df['Line with active holds'])
  if len(lines[0]) != 10:
    return [False]*len(df)

  accept = lambda line: re.search('000....000', line) and any(x in line for x in list('1234'))
  idxs = [i for i, line in enumerate(lines) if accept(line)]
  res = filter_short_runs(idxs, len(lines), GLOBAL_MIN_LINES_LONG)
  return res


def mid6_doubles(df):
  # Note - can be redundant with mid4; modify chart tags accordingly
  lines = list(df['Line with active holds'])
  if len(lines[0]) != 10:
    return [False]*len(df)

  accept = lambda line: re.search('00......00', line) and any(x in line for x in list('1234'))
  idxs = [i for i, line in enumerate(lines) if accept(line)]
  res = filter_short_runs(idxs, len(lines), GLOBAL_MIN_LINES_LONG)
  return res


def filter_short_runs(idxs, n, filt_len):
  # From a list of indices, constructs a list of bools
  # where an index is True only if it is part of a long run
  res = []
  idx_set = set(idxs)
  i = 0
  while i < n:
    if i not in idx_set:
      res.append(False)
      i += 1
    else:
      j = i + 1
      while j in idx_set:
        j += 1

      if j - i >= filt_len:
        res += [True]*(j-i)
      else:
        res += [False]*(j-i)
      i = j
  return res


'''
  Global - line + movement. Needs larger context -- filter out <X in a row
  x	Bracket run: Run (alternating feet, consistent bpm) that includes bracket
  x	Jump run: Many jumps in a row
  Needs additional framework: panel reasoning over downpress lines
  x Single stair
  x Double stair
  x Broken doubles stairs (missing any of the last six notes)
  x Spin (or do with raw lines?)
'''
def bracket_run(df):
  idxs = []
  i = 0
  while i < len(df):
    row = df.iloc[i]
    lf, rf = row['Left foot'], row['Right foot']
    if lf + rf != 1:
      i += 1
      continue
    
    # Find strictly alternating feet sections
    prev_foot = 'Left' if lf else 'Right'
    j = i + 1
    while j < len(df):
      rowj = df.iloc[i]
      lfj, rfj = rowj['Left foot'], rowj['Right foot']
      if lfj + rfj != 1:
        break
      if prev_foot == 'Left' and not rfj:
        break
      if prev_foot == 'Right' and not lfj:
        break
      prev_foot = 'Left' if lfj else 'Right'

    # Filter section: length, consistent bpm, has bracket
    if j - i < GLOBAL_MIN_LINES_LONG:
      i += 1
      continue

    ts = df['Time since'].iloc[i+1:j]
    if len(set(ts)) != 1:
      i += 1
      continue
    
    lines = df['Line'].iloc[i:j]
    has_bracket = lambda line: line.count('1') == 2
    if not any(has_bracket(line) for line in lines):
      i += 1
      continue
    
    for k in range(i, j):
      idxs.append(k)
    i += 1

  res = filter_short_runs(idxs, len(df), GLOBAL_MIN_LINES_LONG)
  return res


def jump_run(df):
  accept = lambda row: row['jump'] > 0
  idxs = [i for i, row in df.iterrows() if accept(row)]
  res = filter_short_runs(idxs, len(df), GLOBAL_MIN_LINES_SHORT)
  return res


def singles_stair(df):
  if len(df['Line'][0]) == 5:
    return singles_stair_in_singles(df)
  else:
    return singles_stair_in_doubles(df)


def singles_stair_in_doubles(df):
  # Uses downpress lines only, compare to panels
  dp_lines = df[df['Has downpress']]['Line with active holds']
  dp_lines = [line.replace('`', '') for line in dp_lines]
  dp_idxs = df[df['Has downpress']].index
  ltr_p1 = ['1000000000', '0100000000', '0010000000', '0001000000', '0000100000']
  rtl_p1 = ['0000100000', '0001000000', '0010000000', '0100000000', '1000000000']
  ltr_p2 = ['0000010000', '0000001000', '0000000100', '0000000010', '0000000001']
  rtl_p2 = ['0000000001', '0000000010', '0000000100', '0000001000', '0000010000']
  patterns = [ltr_p1, rtl_p1, ltr_p2, rtl_p2]
  idxs = []
  for i in range(len(dp_lines) - 5 + 1):
    temp_lines = dp_lines[i:i+5]
    translated_lines = [line.replace('2', '1') for line in temp_lines]
    translated_lines = [line.replace('4', '0') for line in translated_lines]
    if any(translated_lines == x for x in patterns):
      idxs += list(dp_idxs[i:i+5])
  res = filter_short_runs(idxs, len(df), 1)
  return res


def singles_stair_in_singles(df):
  # Uses downpress lines only, compare to panels
  dp_lines = df[df['Has downpress']]['Line with active holds']
  dp_lines = [line.replace('`', '') for line in dp_lines]
  dp_idxs = df[df['Has downpress']].index
  ltr = ['10000', '01000', '00100', '00010', '00001']
  rtl = ltr[::-1]
  idxs = []
  for i in range(len(dp_lines) - 5 + 1):
    temp_lines = dp_lines[i:i+5]
    translated_lines = [line.replace('2', '1') for line in temp_lines]
    translated_lines = [line.replace('4', '0') for line in translated_lines]
    if translated_lines == ltr or translated_lines == rtl:
      idxs += list(dp_idxs[i:i+5])
  res = filter_short_runs(idxs, len(df), 1)
  return res


def doubles_stair(df):
  # Uses downpress lines only, compare to panels
  dp_lines = df[df['Has downpress']]['Line with active holds']
  dp_lines = [line.replace('`', '') for line in dp_lines]
  dp_idxs = df[df['Has downpress']].index
  ltr = ['1000000000', '0100000000', '0010000000', '0001000000', '0000100000',
         '0000010000', '0000001000', '0000000100', '0000000010', '0000000001']
  rtl = ltr[::-1]
  patterns = [ltr, rtl]
  idxs = []
  for i in range(len(dp_lines) - 10 + 1):
    temp_lines = dp_lines[i:i+10]
    translated_lines = [line.replace('2', '1') for line in temp_lines]
    translated_lines = [line.replace('4', '0') for line in translated_lines]
    if any(translated_lines == x for x in patterns):
      idxs += list(dp_idxs[i:i+10])
  res = filter_short_runs(idxs, len(df), 1)
  return res


def doubles_broken_stair(df):
  # Uses downpress lines only, compare to panels
  dp_lines = df[df['Has downpress']]['Line with active holds']
  dp_lines = [line.replace('`', '') for line in dp_lines]
  dp_idxs = df[df['Has downpress']].index
  ltr = ['1000000000', '0100000000', '0010000000', '0001000000', '0000100000',
         '0000010000', '0000001000', '0000000100', '0000000010', '0000000001']
  ltrs = [ltr[:i] + ltr[i+1:] for i in range(4, len(ltr))]
  rtls = [x[::-1] for x in ltrs]
  patterns = ltrs + rtls
  idxs = []
  for i in range(len(dp_lines) - 10 + 1):
    temp_lines = dp_lines[i:i+10]
    translated_lines = [line.replace('2', '1') for line in temp_lines]
    translated_lines = [line.replace('4', '0') for line in translated_lines]
    if any(translated_lines == x for x in patterns):
      idxs += list(dp_idxs[i:i+10])
  res = filter_short_runs(idxs, len(df), 1)
  return res


def spin(df):
  '''
    Based on 5/3-panel spins in singles and 4/6-panel spins in doubles, 
    the maximum angle turned in a single motion is 135
    Most extreme chart is Witch Doctor D19
    - Given two angles, get orientation (clockwise / ccw)
    - Given base angle, query angle and orientation, determine if query angle is in allowed zone (+135 in orientation, -5 in opposite orientation)    
    i, j = base two lines defining orientation
    increment k while query angle is in allowed zone, if total angle surpasses 360 degrees then label as spin.
  '''
  idxs = set()
  angles = list(df['Body angle'])
  i = 0
  while i < len(df) - 1:
    j = i + 1
    spin_orient = get_orientation(angles[i], angles[j])
    total_spin = relative_angle(angles[i], angles[j], spin_orient)

    k = j + 1
    while k < len(df):
      if spin_allowed(angles[k-1], angles[k], spin_orient):
        total_spin += relative_angle(angles[k-1], angles[k], spin_orient)
        k += 1
      else:
        break

      if total_spin >= 350:
        break

    if total_spin >= 350:
      for idx in range(i, k + 1):
        idxs.add(idx)

    i += 1

  res = filter_short_runs(idxs, len(df), 1)
  return res


def relative_angle(angle1, angle2, orient):
  # Relative angle from angle1 to angle2 along orientation (ccw/cw)
  # 1 degree in same orientation = 1, in opposite orientation = 359
  rel_angle = (angle2 - angle1) % 360
  if orient == 'cw':
    return 360 - rel_angle
  else:
    return rel_angle


def get_orientation(angle1, angle2):
  '''
    Angles are in [0, 360]
    Return 'counterclockwise' or 'clockwise': from angle1 to angle2
  '''
  rel_angle = (angle2 - angle1) % 360
  if rel_angle <= 180:
    return 'ccw'
  else:
    return 'cw'


def spin_allowed(base_angle, query_angle, spin_orientation):
  # +135 in orientation, -0 in opposite, with leniency
  leniency = 5
  curr_orient = get_orientation(base_angle, query_angle)
  rel_angle = relative_angle(base_angle, query_angle, curr_orient)
  if curr_orient == spin_orientation and rel_angle <= 135 + leniency:
    return True
  elif curr_orient != spin_orientation and rel_angle >= 360 - leniency:
    return True
  return False


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

  df = pd.read_csv(inp_dir_c + f'{nm} {move_skillset}.csv', index_col=0)
  df = annotate_general(df)
  df = annotate_local(df)
  df = annotate_global(df)
  df.to_csv(out_dir + f'{nm} {move_skillset}.csv')
  return


@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  # nm = 'Super Fantasy - SHK S19 arcade'
  nm = 'Last Rebirth - SHK S15 arcade'
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