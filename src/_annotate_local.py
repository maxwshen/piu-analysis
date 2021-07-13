import numpy as np
import _notelines

mover = None
get_ds = None

GLOBAL_MIN_LINES_SHORT = 4
GLOBAL_MIN_LINES_LONG = 8

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
  '''
    Angle between pads covered between legs
  '''
  # ['none', '90', 'close diagonal', 'far diagonal', '180']
  d1, d2 = get_ds(row1, row2)
  body_angle = row2['Body angle']
  angle = min(body_angle, 360 - body_angle)
  leniency_90 = 15
  leniency_180 = 5
  if angle < 90 - leniency_90:
    return 'none'
  elif 90 - leniency_90 <= angle <= 90 + leniency_90:
    return '90'
  elif 90 + leniency_90 < angle <= 180 - leniency_180:
    lpos = np.array(mover.pos_to_center[d2['limb_to_pos']['Left foot']])
    rpos = np.array(mover.pos_to_center[d2['limb_to_pos']['Right foot']])
    threshold_mm = 250
    if np.linalg.norm(lpos - rpos) < threshold_mm:
      return 'close diagonal'
    else:
      return 'far diagonal'
  else:
    return '180'


def is_footswitch(row1, row2):
  # If exists a pad that was hit by 1 limb, then other limb
  # Require that line1 == line2 and only one downpress
  # More general than movement annotations to capture Native S20
  d1, d2 = get_ds(row1, row2)
  if row1 is None:
    return False
  if row2['jump']:
    return False

  line1 = row1['Line with active holds'].replace('`', '')
  line2 = row2['Line with active holds'].replace('`', '')
  if line1 != line2:
    return False

  if _notelines.num_downpress(line2) != 1:
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

  line1 = row1['Line with active holds'].replace('`', '')
  line2 = row2['Line with active holds'].replace('`', '')
  if line1 != line2:
    return False

  if _notelines.num_downpress(line2) != 1:
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

  line1 = row1['Line with active holds'].replace('`', '')
  line2 = row2['Line with active holds'].replace('`', '')
  if line1 != line2:
    return False

  if _notelines.num_downpress(line2) != 2:
    return False

  old_limbs = ['Left foot', 'Right foot']
  new_limbs = ['Right foot', 'Left foot']
  for old_limb, new_limb in zip(old_limbs, new_limbs):
    old_pos = d1['limb_to_pos'][old_limb]
    new_pos = d1['limb_to_pos'][new_limb]
    if old_pos == new_pos and new_pos in mover.bracket_pos:
      prev_ha = d1['limb_to_heel_action'][old_limb]
      prev_ta = d1['limb_to_toe_action'][old_limb]
      curr_ha = d2['limb_to_heel_action'][new_limb]
      curr_ta = d2['limb_to_toe_action'][new_limb]

      if set([prev_ha, prev_ta, curr_ha, curr_ta]) == set(['1']):
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


#
funcs = {
  # Line only
  'Hold':                 is_hold,
  'Hold taps':            is_hold_taps,
  'Splits':               is_splits,
  # Movement required
  'Jump':                 is_jump,
  'Bracket':              is_bracket,
  'Double step':          is_doublestep,
  'Twist angle':          twist_angle,
  'Footswitch':           is_footswitch,
  'Jack':                 is_jack,
  'Bracket footswitch':   is_bracket_footswitch,
  'Hold tap single foot': is_hold_tap,
  'Hold footslide':       is_hold_footslide,
  'Hold footswitch':      is_hold_footswitch,
  'Staggered hit':        is_multiline,
  'Hands':                is_hands,
  'Travel (mm)':          travel,
}

annot_types = {
  'Twist angle': str,
  'Travel (mm)': float,
}
for a in funcs:
  if a not in annot_types:
    annot_types[a] = bool