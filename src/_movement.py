#
import _data
import _config, _params
from collections import defaultdict, Counter
import numpy as np, pandas as pd
import os, copy
from typing import List, Dict, Set, Tuple

singles_pos_df = pd.read_csv(_config.DATA_DIR + f'positions_singles.csv', index_col = 0)
doubles_pos_df = pd.read_csv(_config.DATA_DIR + f'positions_doubles.csv', index_col = 0)

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
  def __init__(self, style = 'singles', move_skillset = 'default'):
    self.style = style

    if style == 'singles':
      self.df = singles_pos_df
    elif style == 'doubles':
      self.df = doubles_pos_df

    self.costs = _params.movement_costs[move_skillset]

    self.pos_to_cost = {}
    self.pos_to_center = {}
    self.pos_to_heel_coord = {}
    self.pos_to_toe_coord = {}
    self.pos_to_heel_panel = {}
    self.pos_to_toe_panel = {}
    self.pos_to_rotation = {}
    for idx, row in self.df.iterrows():
      nm = row['Name']
      self.pos_to_cost[nm] = row[f'Cost - {move_skillset}']
      self.pos_to_center[nm] = np.array([row['Loc x - center'], row['Loc y - center']])
      self.pos_to_heel_coord[nm] = np.array([row['Loc x - heel ball'], row['Loc y - heel ball']])
      self.pos_to_toe_coord[nm] = np.array([row['Loc x - toe ball'], row['Loc y - toe ball']])
      self.pos_to_heel_panel[nm] = row['Panel - heel']
      self.pos_to_toe_panel[nm] = row['Panel - toe']
      self.pos_to_rotation[nm] = row['Rotation']

    self.all_limbs = ['Left foot', 'Right foot', 'Left hand', 'Right hand']
    self.downpress = set(['1', '2'])
    self.doublestep_prev = set(['1', '2', '3'])
    self.doublestep_curr = set(['1', '2'])
    self.prev_hold = set(['2', '4'])
    self.ok_hold = set(['3', '4'])
    self.verbose = False
    pass


  '''
    Helper
  '''
  def parse_stanceaction(self, sa: str) -> dict:
    '''
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
  def angle_cost(self, d: dict) -> float:
    '''
      Todo -- more, beyond angle? 
      e.g., penalize heel in backswing
    '''
    cost = 0

    # Angle
    left_angle = self.pos_to_rotation[d['limb_to_pos']['Left foot']]
    right_angle = self.pos_to_rotation[d['limb_to_pos']['Right foot']]
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

    if self.verbose: print(f'Angle cost: {cost}')
    return cost


  def foot_inversion_cost(self, d: dict):
    '''
      Penalize if the right foot is left of left foot by a large distance
      Inversion distance threshold = 185 mm (distance from center to corner). Basically only penalize 180 twists. Penalty should be less than a double step so that we prefer to do a 180 twist than double step once, but prefer to double step to avoid multiple 180 twists
    '''
    cost = 0
    left_coord = self.pos_to_center[d['limb_to_pos']['Left foot']]
    right_coord = self.pos_to_center[d['limb_to_pos']['Right foot']]

    diff = left_coord[0] - right_coord[0]

    if 0 < diff <= self.costs['Inversion distance threshold']:
      cost += self.costs['Inverted feet small']

    if diff > self.costs['Inversion distance threshold']:
      cost += self.costs['Inverted feet big']

    if self.verbose: print(f'Foot inversion cost: {cost}')
    return cost


  def hold_change_cost(self, d1: dict, d2: dict):
    '''
      Penalize switching limb/part on a hold
    '''
    cost = 0
    for limb in d2['limb_to_pos']:
      if limb not in d1['limb_to_pos']:
        continue

      prev_ha = d1['limb_to_heel_action'][limb]
      prev_ta = d1['limb_to_toe_action'][limb]

      curr_ha = d2['limb_to_heel_action'][limb]
      curr_ta = d2['limb_to_toe_action'][limb]

      if prev_ha == '4' and curr_ha != '4':
        if curr_ta == '4':
          cost += self.costs['Hold footslide']

      if prev_ta == '4' and curr_ta != '4':
        if curr_ha == '4':
          cost += self.costs['Hold footslide']

      if prev_ha in self.prev_hold or prev_ta in self.prev_hold:
        if curr_ta not in self.ok_hold and curr_ha not in self.ok_hold:
          cost += self.costs['Hold footswitch']

    if self.verbose: print(f'Hold change cost: {cost}')
    return cost


  def foot_pos_cost(self, d: dict) -> float:
    '''
      Sum over limbs
    '''
    cost = 0
    for limb in d['limb_to_pos']:
      pos = d['limb_to_pos'][limb]
      cost += self.pos_to_cost[pos]
    if self.verbose: print(f'Pos cost: {cost}')
    return cost


  def move_cost(self, d1: dict, d2: dict, time = None) -> float:
    '''
      Sum over limbs, distance moved by
      - foot center
      - average of heel and toe
    '''
    cost = 0
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
      dist = np.mean(dist_heel + dist_toe)

      cost += dist / self.costs['Distance normalizer']

    if time is not None:
      if 0.001 < time < self.costs['Time threshold']:
        '''
          Ex. normalizer = 300 ms, then
          400 ms since = 3/4 cost
          100 ms since = 3 cost
        '''
        time_factor = self.costs['Time normalizer'] / time
        cost *= time_factor

    if cost == 0:
      cost = self.costs['No movement reward']

    if self.verbose: print(f'Move cost: {cost}')
    return cost


  def double_step_cost(self, d1: dict, d2: dict, time = None) -> float:
    '''
      Indirectly reward longer time since last foot movement. Cannot directly penalize by time since last foot movement in current graph representation

      Add cost only when a single foot is used twice, and only one foot is used both times, and no limb is in a hold
    '''
    cost = 0
    num_limbs_doubling = 0
    num_limbs_prev = 0
    num_limbs_now = 0
    num_limbs_hold = 0
    for limb in d2['limb_to_pos']:
      if limb not in d1['limb_to_heel_action']:
        continue

      prev_heel = d1['limb_to_heel_action'][limb] in self.doublestep_prev
      prev_toe = d1['limb_to_toe_action'][limb] in self.doublestep_prev
      curr_heel = d2['limb_to_heel_action'][limb] in self.doublestep_curr
      curr_toe = d2['limb_to_toe_action'][limb] in self.doublestep_curr

      prev_step = prev_heel or prev_toe
      curr_step = curr_heel or curr_toe

      if prev_step and curr_step:
        num_limbs_doubling += 1
      if prev_heel or prev_toe:
        num_limbs_prev += 1
      if curr_heel or curr_toe:
        num_limbs_now += 1

      curr_heel_hold = d2['limb_to_heel_action'][limb] in self.ok_hold
      curr_toe_hold = d2['limb_to_toe_action'][limb] in self.ok_hold
      if curr_heel_hold or curr_toe_hold:
        num_limbs_hold += 1

    if num_limbs_doubling == 1 and num_limbs_prev == 1 and num_limbs_now == 1 and num_limbs_hold == 0:
      cost += self.costs['Double step']

    if time is not None:
      if 0.001 < time < self.costs['Time threshold']:
        '''
          Ex. normalizer = 300 ms, then
          400 ms since = 3/4 cost
          100 ms since = 3 cost
        '''
        # time_factor = self.costs['Time normalizer'] / time
        time_factor = 1
        cost *= time_factor
      elif time >= self.costs['Time threshold']:
        cost = 0

    if self.verbose: print(f'Double step cost: {cost}')
    return cost


  def jump_cost(self, d1: dict, d2: dict) -> float:
    '''
      Penalize if both feet move
    '''
    cost = 0
    prev_left = d1['limb_to_pos']['Left foot']
    prev_right = d1['limb_to_pos']['Right foot']
    curr_left = d2['limb_to_pos']['Left foot']
    curr_right = d2['limb_to_pos']['Right foot']
    if prev_left != curr_left and prev_right != curr_right:
      cost += self.costs['Jump']

    if self.verbose: print(f'Jump cost: {cost}')
    return cost


  def bracket_cost(self, d: dict) -> float:
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


  def hands_cost(self, d: dict) -> float:
    '''
      Sum over limbs
    '''
    cost = 0
    if len(d['limb_to_pos']) > 2:
      cost += self.costs['Hands']
    if self.verbose: print(f'Hands cost: {cost}')
    return cost


  def move_without_action_cost(self, d1: dict, d2: dict) -> float:
    cost = 0
    for limb in ['Left foot', 'Right foot']:
      prev_pos = d1['limb_to_pos'][limb]
      curr_pos = d2['limb_to_pos'][limb]

      heel_action = d2['limb_to_heel_action'][limb]
      toe_action = d2['limb_to_toe_action'][limb]
      no_action = bool(heel_action + toe_action == '--')

      if prev_pos != curr_pos and no_action:
        cost += self.costs['Move without action']

    if self.verbose: print(f'Move without action cost: {cost}')
    return cost


  def downpress_cost(self, d: dict) -> float:
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
  def get_cost_from_ds(self, d1: dict, d2: dict, time: float = 1, verbose: bool = False) -> float:
    '''
    '''
    self.verbose = verbose

    cost = 0
    cost += self.angle_cost(d2)
    cost += self.foot_inversion_cost(d2)
    cost += self.foot_pos_cost(d2)
    cost += self.hold_change_cost(d1, d2)
    # mv_cost = self.move_cost(d1, d2, time)
    mv_cost = self.move_cost(d1, d2)
    cost += mv_cost
    cost += self.jump_cost(d1, d2)
    cost += self.bracket_cost(d2)
    cost += self.hands_cost(d2)
    cost += self.move_without_action_cost(d1, d2)
    cost += self.downpress_cost(d2)

    '''
      Conditional costs

      If movement: apply double step cost
      If no movement:
        If time slower than 270 npm, no cost for jacks
        else: apply high cost to get footswitching
    '''
    ds_cost = self.double_step_cost(d1, d2, time)
    if mv_cost >= 0:
      cost += ds_cost
    elif mv_cost <= 0:
      if time <= _params.jacks_footswitch_t_thresh:
        cost += ds_cost

    return cost


  def get_cost_from_text(self, sa1: str, sa2: str, time: float = 1, verbose: bool = False) -> float:
    '''
    '''
    self.verbose = verbose

    d1 = self.parse_stanceaction(sa1)
    d2 = self.parse_stanceaction(sa2)
    return self.get_cost_from_ds(d1, d2, time = time, verbose = verbose)


  '''
    Heuristic node/edge pruning
  '''
  def unnecessary_jump(self, d1: dict, d2: dict, line: str) -> bool:
    '''
      Detect if unnecessary jump
    '''
    num_nonpresses = line.count('0')
    if num_nonpresses == len(line) - 1:
      for downpress in self.downpress:
        if downpress in line:
          if self.jump_cost(d1, d2):
            return True
    return False


  def beginner_ok(self, d2: dict) -> bool:
    '''
      Filter all stances other than air-X
    '''
    for limb in d2['limb_to_pos']:
      pos = d2['limb_to_pos'][limb]
      if 'a' not in pos:
        return False
    return True


  def call_twist(self, sa: str, lwah: str):
    '''
      Label type of twist
      - 90 degree
      - Diagonal
      - 180 degree
      Facing left or right
    '''
    d = self.parse_stanceaction(sa)


    return


  def multihit_modifier(self, d1: dict, d2: dict, node_nm: str):
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

'''
  Testing
'''
def test():
  mover = Movement(style = 'singles')

  # test_basic(mover)
  test_holds(mover)

  return


def test_basic(mover):
  # 00000 -> 01000
  sa1 = '14,36;--,--'
  sa2s = [
    '47,36;-1,--',
    '57,36;-1,--',
    'a9,57;--,-1',
  ]
  for sa2 in sa2s:
    cost = mover.get_cost(sa1, sa2, verbose = True)
    print(sa2, cost, '\n')
  return


def test_holds(mover):
  # 40001 -> 40100
  sa1 = '14,36;4-,1-'
  sa2s = [
    '14,56;4-,1-',
    '14,54;4-,1-',
    'a1,56;-4,1-',
    '54,a1;1-,-4',
  ]
  for sa2 in sa2s:
    cost = mover.get_cost(sa1, sa2, verbose = True)
    print(sa2, cost, '\n')
  return


if __name__ == '__main__':
  test()