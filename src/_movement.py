#
import _data
import _config, _params
from collections import defaultdict, Counter, namedtuple
import numpy as np, pandas as pd
import os, copy
from typing import List, Dict, Set, Tuple

import _positions, _notelines, _graph, _stepcharts


scinfo = _stepcharts.SCInfo()


def nm_to_moveskillset(nm):
  level = scinfo.name_to_level[nm]
  if level <= 11:
    return 'beginner'
  else:
    return 'basic'  

'''
  Movement
'''
class Movement():
  '''
    Calculates the cost between two stanceactions

    Cost values are obtained from:
      _params.py
      positions_<singles/doubles>.csv
  '''
  def __init__(self, style = 'singles', move_skillset = 'basic', custom_cost = None):
    self.style = style
    self.move_skillset = move_skillset

    if style == 'singles':
      self.df = _positions.singles_pos_df
      panel_cols = [
        'p1,1', 'p1,3', 'p1,5', 'p1,7', 'p1,9',
      ]
    elif style == 'doubles':
      self.df = _positions.doubles_pos_df
      panel_cols = [
        'p1,1', 'p1,3', 'p1,5', 'p1,7', 'p1,9',
        'p2,1', 'p2,3', 'p2,5', 'p2,7', 'p2,9',
      ]

    self.params = _params.movement_costs[move_skillset]['parameters']
    if custom_cost:
      self.costs = custom_cost
    else:
      self.costs = _params.movement_costs[move_skillset]['costs']

    self.min_cost = sum(cost for cost in self.costs.values() if cost < 0)

    self.pos_to_cost = {}
    self.pos_to_center = {}
    self.pos_to_heel_coord = {}
    self.pos_to_toe_coord = {}
    self.pos_to_heel_panel = {}
    self.pos_to_toe_panel = {}
    self.pos_to_rotation = {}
    self.bracket_pos = set()
    for idx, row in self.df.iterrows():
      nm = row['Name']
      self.pos_to_cost[nm] = row[f'Cost - {move_skillset}']
      self.pos_to_center[nm] = np.array([row['Loc x - center'], row['Loc y - center']])
      self.pos_to_heel_coord[nm] = np.array([row['Loc x - heel ball'], row['Loc y - heel ball']])
      self.pos_to_toe_coord[nm] = np.array([row['Loc x - toe ball'], row['Loc y - toe ball']])
      self.pos_to_heel_panel[nm] = row['Panel - heel']
      self.pos_to_toe_panel[nm] = row['Panel - toe']
      self.pos_to_rotation[nm] = row['Rotation']
      if sum(row[panel_cols]) > 1:
        self.bracket_pos.add(nm)

    self.all_limbs = ['Left foot', 'Right foot', 'Left hand', 'Right hand']
    self.downpress = set(list('12'))
    self.doublestep_prev = set(list('123'))
    self.doublestep_curr = set(list('12'))
    self.prev_hold = set(list('24'))
    self.ok_hold = set(list('34'))
    self.verbose = False
    pass


  '''
    Helper
  '''
  def parse_stanceaction(self, sa):
    '''
      TODO - Consider moving LRU cache from c_dijkstra to here?
    '''
    [stance, action] = sa.split(';')
    limb_to_pos = {limb: pos for limb, pos in zip(self.all_limbs, stance.split(','))}

    limb_to_heel_action = {}
    limb_to_toe_action = {}
    for limb, a in zip(self.all_limbs, action.split(',')):
      heel_action, toe_action = a[0], a[1]
      limb_to_heel_action[limb] = heel_action
      limb_to_toe_action[limb] = toe_action

    return {
      'limb_to_pos': limb_to_pos,
      'limb_to_heel_action': limb_to_heel_action,
      'limb_to_toe_action': limb_to_toe_action,
    }


  '''
    Costs
  '''
  def angle_cost(self, d, inv_cost):
    cost = 0

    left_angle = self.pos_to_rotation[d['limb_to_pos']['Left foot']]
    right_angle = self.pos_to_rotation[d['limb_to_pos']['Right foot']]

    # Not inverted
    if inv_cost == 0:
      if left_angle == 'any' or right_angle == 'any':
        pass
      else:
        left_angle = float(left_angle)
        right_angle = float(right_angle)
    
        angle = -1 * left_angle + right_angle
        if angle > 170:
          cost += self.costs['Angle too open']
        if angle < 0:
          cost += self.costs['Angle duck']
        if angle < -45:
          cost += self.costs['Angle extreme duck']
    elif inv_cost > 0:
      if left_angle != 'any' and right_angle != 'any':
        cost += self.costs['Angle non-air inverted']

    if self.verbose: print(f'Angle cost: {cost}')
    return cost


  def foot_inversion_cost(self, d):
    '''
      Penalize if the right foot is left of left foot by a large distance
      Inversion distance threshold = 185 mm (distance from center to corner). Basically only penalize 180 twists. Penalty should be less than a double step so that we prefer to do a 180 twist than double step once, but prefer to double step to avoid multiple 180 twists
    '''
    cost = 0
    left_coord = self.pos_to_center[d['limb_to_pos']['Left foot']]
    right_coord = self.pos_to_center[d['limb_to_pos']['Right foot']]

    diff = left_coord[0] - right_coord[0]

    if 0 < diff <= self.params['Inversion distance threshold']:
      cost += self.costs['Inverted feet small']

    if diff > self.params['Inversion distance threshold']:
      cost += self.costs['Inverted feet big']

    if self.verbose: print(f'Foot inversion cost: {cost}')
    return cost


  def hand_inversion_cost(self, d):
    '''
      Penalize if the right hand is left of left hand
    '''
    cost = 0
    if len(d['limb_to_pos']) <= 2:
      return 0

    left_coord = self.pos_to_center[d['limb_to_pos']['Left hand']]
    right_coord = self.pos_to_center[d['limb_to_pos']['Right hand']]
    diff = left_coord[0] - right_coord[0]
    if 0 < diff:
      cost += self.costs['Inverted hands']

    if self.verbose: print(f'Hand inversion cost: {cost}')
    return cost


  def hold_footswitch_cost(self, d1, d2):
    '''
      Penalize switching limb/part on a hold
    '''
    cost = 0
    for limb in d2['limb_to_pos']:
      if limb in d1['limb_to_pos']:
        prev_ha = d1['limb_to_heel_action'][limb]
        prev_ta = d1['limb_to_toe_action'][limb]
      else:
        prev_ha = '-'
        prev_ta = '-'

      curr_ha = d2['limb_to_heel_action'][limb]
      curr_ta = d2['limb_to_toe_action'][limb]

      if prev_ha in self.prev_hold or prev_ta in self.prev_hold:
        if curr_ta not in self.ok_hold and curr_ha not in self.ok_hold:
          cost += self.costs['Hold footswitch']

      if prev_ha not in self.prev_hold and prev_ta not in self.prev_hold:
        if curr_ta == '3' or curr_ha == '3':
          cost += 100

    if self.verbose: print(f'Hold footswitch cost: {cost}')
    return cost


  def hold_footslide_cost(self, d1, d2):
    '''
      Penalize switching limb/part on a hold
    '''
    cost = 0
    for limb in d2['limb_to_pos']:
      if limb in d1['limb_to_pos']:
        prev_ha = d1['limb_to_heel_action'][limb]
        prev_ta = d1['limb_to_toe_action'][limb]
      else:
        prev_ha = '-'
        prev_ta = '-'

      curr_ha = d2['limb_to_heel_action'][limb]
      curr_ta = d2['limb_to_toe_action'][limb]

      if prev_ha == '4' and curr_ha != '4':
        if curr_ta == '4':
          cost += self.costs['Hold footslide']

      if prev_ta == '4' and curr_ta != '4':
        if curr_ha == '4':
          cost += self.costs['Hold footslide']

    if self.verbose: print(f'Hold footslide cost: {cost}')
    return cost


  def foot_pos_cost(self, d):
    '''
      Sum over limbs
    '''
    cost = 0
    for limb in d['limb_to_pos']:
      pos = d['limb_to_pos'][limb]
      cost += self.pos_to_cost[pos]
    if self.verbose: print(f'Pos cost: {cost}')
    return cost


  def move_cost(self, d1, d2):
    '''
      Sum over limbs, distance moved by
      - foot center
      - average of heel and toe

      Do not grant no-movement reward when alternating heel and toe on same foot
    '''
    cost = 0
    has_bracket = False
    for limb in d2['limb_to_pos']:
      if limb not in d1['limb_to_pos']:
        continue

      '''
        Center to center
      '''
      # prev_center = self.pos_to_center[d1['limb_to_pos'][limb]]
      # new_center = self.pos_to_center[d2['limb_to_pos'][limb]]
      # dist = np.linalg.norm(prev_center - new_center, ord = 2)

      '''
        Avg of heel to heel and toe to toe
      '''
      prev_heel = self.pos_to_heel_coord[d1['limb_to_pos'][limb]]
      new_heel = self.pos_to_heel_coord[d2['limb_to_pos'][limb]]
      prev_toe = self.pos_to_toe_coord[d1['limb_to_pos'][limb]]
      new_toe = self.pos_to_toe_coord[d2['limb_to_pos'][limb]]
      dist_heel = np.linalg.norm(prev_heel - new_heel, ord = 2)
      dist_toe = np.linalg.norm(prev_toe - new_toe, ord = 2)
      # dist = np.mean(dist_heel + dist_toe)
      dist = np.min([dist_heel, dist_toe])

      cost += dist / self.params['Distance normalizer']

      if d2['limb_to_pos'][limb] in self.bracket_pos:
        has_bracket = True

      # Do not grant no-movement reward when alternating heel and toe on same foot
      if any(x in d1['limb_to_heel_action'][limb] for x in list('12')):
        if any(x in d2['limb_to_toe_action'][limb] for x in list('12')):
          cost += self.costs['Toe-heel alternate']

      if any(x in d1['limb_to_toe_action'][limb] for x in list('12')):
        if any(x in d2['limb_to_heel_action'][limb] for x in list('12')):
          cost += self.costs['Toe-heel alternate']

    # if has_bracket and cost == 0:
    #   cost = 0.01

    if cost == 0:
      cost = self.costs['No movement reward']

    if cost > 0:
      cost = cost ** self.costs['Move power']

    if self.verbose: print(f'Move cost: {cost}')
    return cost


  def double_step_cost(self, d1, d2):
    '''
      Straightforward definition of double step: A limb is used on two neighboring lines.
      Forgive:
      - double step on same panel, going from 1/3 -> 1/2
      - double steps in active holds
      - double steps with repeated stance-action
    '''
    cost = 0
    limbs = list(limb for limb in d2['limb_to_pos']
                      if limb in d1['limb_to_heel_action'])

    def has_active_hold(limb):
      heel = any(x in d2['limb_to_heel_action'][limb] for x in list('34'))
      toe = any(x in d2['limb_to_toe_action'][limb] for x in list('34'))
      return heel or toe

    active_hold = any(has_active_hold(limb) for limb in limbs)
    if not active_hold:
      for limb in limbs:
        prev_heel = d1['limb_to_heel_action'][limb] in self.doublestep_prev
        prev_toe = d1['limb_to_toe_action'][limb] in self.doublestep_prev
        curr_heel = d2['limb_to_heel_action'][limb] in self.doublestep_curr
        curr_toe = d2['limb_to_toe_action'][limb] in self.doublestep_curr

        prev_step = prev_heel or prev_toe
        curr_step = curr_heel or curr_toe

        limb_cost = 0
        if prev_step and curr_step:
          limb_cost = self.costs['Double step']

        # Forgive 1/3->2/1 with same limb on same pad
        prev_heel_a = d1['limb_to_heel_action'][limb]
        curr_heel_a = d2['limb_to_heel_action'][limb]
        prev_toe_a = d1['limb_to_toe_action'][limb]
        curr_toe_a = d2['limb_to_toe_action'][limb]
        heel_pad_match = d1['limb_to_pos'][limb] == d2['limb_to_pos'][limb]
        toe_pad_match = d1['limb_to_pos'][limb] == d2['limb_to_pos'][limb]
        if prev_heel_a in list('13') and heel_pad_match:
          if any(x in curr_heel_a for x in list('12')):
            limb_cost = 0
        elif prev_toe_a in list('13') and toe_pad_match:
          if any(x in curr_toe_a for x in list('12')):
            limb_cost = 0

        cost += limb_cost

    # Nested equality
    def identical_ds(d1, d2):
      for k in d2.keys():
        if d2[k].items() != d1[k].items():
          return False
      return True

    if identical_ds(d1, d2):
      cost = -1 * self.costs['No movement reward']
      cost += self.costs['Jacks']

    if self.verbose: print(f'Double step cost: {cost}')
    return cost


  def jump_cost(self, d1, d2):
    '''
      Jump: Both feet change position or using both feet
      E.g., not a jump is if 1 foot has the same position and action as before.
    '''
    cost = 0
    feet = ['Left foot', 'Right foot']

    matched_keys = ['limb_to_pos']
    matches = lambda x: all([d1[mkey][x] == d2[mkey][x] for mkey in matched_keys])
    feet_stay = [limb for limb in feet if matches(limb)]
    both_feet_moved = bool(len(feet_stay) == 0)

    pressing_keys = ['limb_to_heel_action', 'limb_to_toe_action']
    pressing = lambda x: any([d2[pkey][x] == '1' or d2[pkey][x] == '2'
                              for pkey in pressing_keys])
    feet_pressing = [limb for limb in feet if pressing(limb)]

    both_feet_pressing = bool(len(feet_pressing) == 2)
    moved = self.move_cost(d1, d2) > 0
    both_feet_pressing_and_moved = both_feet_pressing and moved

    if both_feet_moved or both_feet_pressing:
      cost += self.costs['Jump']

    if self.verbose: print(f'Jump cost: {cost}')
    return cost


  def bracket_cost(self, d):
    '''
      Sum over limbs
    '''
    cost = 0
    for limb in d['limb_to_pos']:
      if d['limb_to_heel_action'][limb] in self.downpress:
        if d['limb_to_toe_action'][limb] in self.downpress:
          cost += self.costs['Bracket']
    if self.verbose: print(f'Bracket cost: {cost}')
    return cost


  def hands_cost(self, d):
    '''
      Sum over limbs
    '''
    cost = 0
    if len(d['limb_to_pos']) > 2:
      cost += self.costs['Hands']
    if self.verbose: print(f'Hands cost: {cost}')
    return cost


  def move_without_action_cost(self, d1, d2):
    cost = 0
    for limb in ['Left foot', 'Right foot']:
      prev_pos = d1['limb_to_pos'][limb]
      curr_pos = d2['limb_to_pos'][limb]

      heel_action = d2['limb_to_heel_action'][limb]
      toe_action = d2['limb_to_toe_action'][limb]
      acts = heel_action + toe_action
      no_action = not any(x in acts for x in list('12'))

      if prev_pos != curr_pos and no_action:
        cost += self.costs['Move without action']

    if self.verbose: print(f'Move without action cost: {cost}')
    return cost


  def downpress_cost(self, d):
    cost = 0
    for limb in ['Left foot', 'Right foot']:
      heel_action = d['limb_to_heel_action'][limb]
      toe_action = d['limb_to_toe_action'][limb]

      heel_press = heel_action in self.downpress
      toe_press = toe_action in self.downpress

      if heel_press or toe_press:
        cost += self.costs['Downpress cost per limb']

    if self.verbose: print(f'Downpress cost: {cost}')
    return cost


  '''
    Primary
  '''
  def get_cost_from_ds(self, d1, d2, verbose = False):
    '''
    '''
    self.verbose = verbose

    inv_cost = self.foot_inversion_cost(d2)
    mv_cost = self.move_cost(d1, d2)
    ds_cost = self.double_step_cost(d1, d2)
    cost = {
      'foot_inversion'      : inv_cost,
      'angle'               : self.angle_cost(d2, inv_cost),
      'hand_inversion'      : self.hand_inversion_cost(d2),
      'foot_position'       : self.foot_pos_cost(d2),
      'hold_footslide'      : self.hold_footslide_cost(d1, d2),
      'hold_footswitch'     : self.hold_footswitch_cost(d1, d2),
      'move'                : mv_cost,
      'jump'                : self.jump_cost(d1, d2),
      'bracket'             : self.bracket_cost(d2),
      'hands'               : self.hands_cost(d2),
      'move_without_action' : self.move_without_action_cost(d1, d2),
      'downpress'           : self.downpress_cost(d2),
      'double_step'         : ds_cost,
      'min_cost'            : -1 * self.min_cost,
    }
    return cost


  def get_cost_from_text(self, sa1, sa2, verbose = False):
    '''
    '''
    self.verbose = verbose

    d1 = self.parse_stanceaction(sa1)
    d2 = self.parse_stanceaction(sa2)
    return self.get_cost_from_ds(d1, d2, verbose=verbose)


  '''
    Heuristic node/edge pruning
  '''
  def unnecessary_jump(self, d1, d2, line):
    # Detect if unnecessary jump
    has_one_press = lambda line: line.count('0') == len(line) - 1
    has_downpress = lambda line: any([dp in line for dp in self.downpress])
    if has_one_press(line) and has_downpress(line):
      if self.jump_cost(d1, d2):
        return True
    return False


  def beginner_ok(self, d2):
    '''
      Filter all stances other than air-X
    '''
    for limb in d2['limb_to_pos']:
      pos = d2['limb_to_pos'][limb]
      if 'a' not in pos:
        return False
    return True


  '''
    Cost modifiers using line node
  '''
  def multihit_modifier(self, d1, d2, node_nm):
    '''
      Apply multihit reward only if brackets are involved
      Remove jump penalty if applied
    '''
    cost = 0
    is_multi = bool('multi' in node_nm)
    if not is_multi:
      return 0

    has_bracket = False
    for limb in d2['limb_to_pos']:
      if d2['limb_to_heel_action'][limb] in self.downpress:
        if d2['limb_to_toe_action'][limb] in self.downpress:
          has_bracket = True
          break

    if has_bracket:
      cost += self.costs['Multi reward']

    jc = self.jump_cost(d1, d2)
    cost -= jc
    return cost


  def bracket_on_singlepanel_line(self, d, line):
    num_downpress = _notelines.num_downpress(line)
    if num_downpress != 1:
      return 0

    cost = 0
    for limb in d['limb_to_pos']:
      pos = d['limb_to_pos'][limb]
      if pos in self.bracket_pos:
        cost += self.costs['Bracket on 1panel line']
    return cost


  def hold_alternate(self, tag1, tag2, motif_len):
    # Apply only once
    # curr_motif is either ((start, end), motif_name) or None
    cost = 0
    if motif_len:
      _, _, tag1_hold = _graph.parse_tag(tag1)
      _, _, tag2_hold = _graph.parse_tag(tag2)
      if tag2_hold == 'alternate' and tag1_hold != 'alternate':
        if motif_len <= _params.hold_tap_line_threshold:
          cost += self.costs['Hold alternate feet for hits (onetime, short)']
        else:
          cost += self.costs['Hold alternate feet for hits (onetime, long)']
      elif tag2_hold == 'free' and tag1_hold != 'free':
        if motif_len <= _params.hold_tap_line_threshold:
          cost += self.costs['Hold free feet for hits (onetime, short)']
        else:
          cost += self.costs['Hold free feet for hits (onetime, long)']        

    return cost


'''
  Testing
'''
def test_singles():
  mover = Movement(style='singles')

  # test_singles_basic(mover)
  # test_singles_holds(mover)
  test_double_step(mover)
  return


def test_singles_basic(mover):
  # 00000 -> 01000
  sa1 = '14,36;--,--'
  sa2s = [
    '47,36;-1,--',
    '57,36;-1,--',
    'a9,57;--,-1',
  ]
  for sa2 in sa2s:
    cost = mover.get_cost_from_text(sa1, sa2, verbose=True)
    print(sa2, cost, '\n')
  return


def test_singles_holds(mover):
  # 40001 -> 40100
  sa1 = '14,36;4-,1-'
  sa2s = [
    '14,56;4-,1-',
    '14,54;4-,1-',
    'a1,56;-4,1-',
    '54,a1;1-,-4',
  ]
  for sa2 in sa2s:
    cost = mover.get_cost_from_text(sa1, sa2, verbose=True)
    print(sa2, cost, '\n')
  return


def test_double_step(mover):
  sas = [
    ('14,36;--,3-', '14,36;--,2-'),
    ('14,36;--,3-', '14,69;--,-1'),
  ]
  for sa1, sa2 in sas:
    cost = mover.get_cost_from_text(sa1, sa2, verbose=True)
    print(sa1, sa2, cost, '\n')
  return


def test_doubles():
  mover = Movement(style='doubles')

  # test_doubles_basic(mover)
  test_doubles_hold_taps(mover)
  return


def test_doubles_basic(mover):
  # #
  # sa1 = 'p1`36c,p2`14c;1-,--'
  # sa2s = [
  #   'p1`36c,p1`a9c;--,-1',
  #   'p1`36c,p2`4-p1`9c;--,-1',
  # ]
  # sa1 = 'p1`69c,p1`3-p2`4c;--,2-'
  sa1 = 'p1`69c,p1`3-p2`1;--,2-'
  sa2s = [
    'p1`69c,p1`3-p2`1;--,41',
  ]

  for sa2 in sa2s:
    cost = mover.get_cost_from_text(sa1, sa2, time=0.25, verbose=True)
    print(sa2, cost, '\n')

    cost = mover.get_cost_from_text(sa1, sa2, time=0.20, verbose=True)
    print(sa2, cost, '\n')
  return


def test_doubles_hold_taps(mover):
  paths = {
    # 'wrong': [
    #   'p1`47,p1`36c;-2,--',
    #   'p1`57,p1`36c;14,--',
    #   'p1`57,p1`36c;-4,1-',
    #   'p1`57,p1`36c;14,--',
    #   'p1`47,p1`36c;-3,--',
    # ],
    # 'right': [
    #   'p1`47,p1`36c;-2,--',
    #   'p1`47,p1`a5;-4,-1',
    #   'p1`47,p1`a3;-4,-1',
    #   'p1`47,p1`a5;-4,-1',
    #   'p1`47,p1`a5;-3,--',
    # ],
    'wrong2': [
      'p1`35,p2`15;2-,1-',
      'p1`36,p2`15;4-,-1',
      'p1`36,p2`14;4-,1-',
      'p1`36,p2`47;4-,-1',
      'p1`36,p2`14c;4-,1-',
      'p1`36,p2`4-p1`9c;4-,-1',
      'p1`36,p2`14c;4-,1-',
      'p1`36,p1`56;4-,1-',
      'p1`36,p2`14c;4-,1-',
      'p1`36,p2`4-p1`9c;4-,-1',
      'p1`36,p2`14c;4-,1-',
      'p1`36,p2`47;4-,-1',
    ],
    'right2': [
      'p1`3-p2`1,p2`a5;21,--',
      'p1`3-p2`1,p2`57;4-,1-',
      'p1`3-p2`1,p2`57;41,--',
      'p1`3-p2`1,p2`47c;4-,-1',
      'p1`3-p2`1,p2`47c;41,--',
      'p1`3-p2`1,p1`69c;4-,-1',
      'p1`3-p2`1,p1`69c;41,--',
      'p1`3-p2`1,p1`59;4-,1-',
      'p1`3-p2`1,p1`59;41,--',
      'p1`3-p2`1,p2`7-p1`9;4-,-1',
      'p1`3-p2`1,p2`7-p1`9;41,--',
      'p1`3-p2`1,p2`a7;4-,-1',
    ],
    # 'wrong3': [
    #   'p2`1-p1`6c,p2`4-p1`9c;--,-2',
    #   'p1`6-p2`7c,p2`4-p1`9c;-1,-4',
    #   'p1`59,p2`4-p1`9c;1-,-4',
    #   'p2`54,p2`4-p1`9c;1-,-4',
    #   'p2`54,p2`4-p1`9c;--,-3',
    # ],
    # 'right3': [
    #   'p1`69,p2`14;-2,--',
    #   'p1`69,p2`47;-4,-1',
    #   'p1`59,p2`47;14,--',
    #   'p1`59,p2`a5;-4,-1',
    #   'p1`69c,p2`a5;-3,--',
    # ]
  }
  for name, path in paths.items():
    total_cost_d = defaultdict(lambda: 0)
    for sa1, sa2 in zip(path[:-1], path[1:]):
      # print('\n', sa1, sa2)
      # cost_d = mover.get_cost_from_text(sa1, sa2, verbose=True)
      cost_d = mover.get_cost_from_text(sa1, sa2)
      for k, v in cost_d.items():
        total_cost_d[k] += v
    cost = sum(total_cost_d.values())
    print('\n', name, cost)
    for k, v in total_cost_d.items():
      print(k.ljust(15), v)

  return


if __name__ == '__main__':
  # test_singles()
  test_doubles()
