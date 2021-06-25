import numpy as np

mover = None
get_ds = None

GLOBAL_MIN_LINES_SHORT = 4
GLOBAL_MIN_LINES_LONG = 8

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
  # Pretty hacky, redo?
  idxs = set()
  for i in range(1, len(df)):
    row1, row2 = df.iloc[i-1], df.iloc[i]
    line1, line2 = row1['Line'], row2['Line']
    if row2['Annotation'] == 'alternate' and '2' in line2:
      idxs.add(i-1)
      idxs.add(i)
    if line1.replace('1', '2') == line2 and '2' in line2:
      idxs.add(i-2)
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


#
funcs = {
  # Line only
  'Run':                    run,
  'Hold run':               hold_run,
  'Drill':                  drill,
  'Bracket drill':          bracket_drill,
  'Irregular rhythm':       irregular_rhythm,
  'Bracket jump run':       bracket_jump_run,
  'Side3 singles':          side3_singles,
  'Mid4 doubles':           mid4_doubles,
  'Mid6 doubles':           mid6_doubles,
  # Movement required
  'Run with brackets':      bracket_run,
  'Jump run':               jump_run,
  'Stairs, singles':        singles_stair,
  'Stairs, doubles':        doubles_stair,
  'Broken stairs, doubles': doubles_broken_stair,
  'Spin':                   spin,
}
annot_types = {}
for a in funcs:
  if a not in annot_types:
    annot_types[a] = bool