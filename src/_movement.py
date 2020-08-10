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
    self.pos_to_rotation = {}
    for idx, row in self.df.iterrows():
      nm = row['Name']
      self.pos_to_cost[nm] = row[f'Cost - {move_skillset}']
      self.pos_to_center[nm] = np.array([row['Loc x - center'], row['Loc y - center']])
      self.pos_to_rotation[nm] = row['Rotation']

    self.all_limbs = ['Left foot', 'Right foot', 'Left hand', 'Right hand']
    self.downpress = ['1', '2']
    self.doublestep_actions = ['1', '2', '3']
    self.verbose = False
    pass

  '''
    Helper
  '''
  def parse_stanceaction(self, sa: str) -> dict:
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
      Penalize if the right foot is left of left foot
    '''
    cost = 0
    left_coord = self.pos_to_center[d['limb_to_pos']['Left foot']]
    right_coord = self.pos_to_center[d['limb_to_pos']['Right foot']]

    if right_coord[0] < left_coord[0]:
      cost += self.costs['Inverted feet']

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

      ok = ['3', '4']
      if prev_ha == '4' or prev_ta == '4':
        if curr_ta not in ok and curr_ha not in ok:
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


  def move_cost(self, d1: dict, d2: dict, time: float) -> float:
    '''
      Sum over limbs, distance moved by center of limb
    '''
    cost = 0
    for limb in d2['limb_to_pos']:
      if limb not in d1['limb_to_pos']:
        continue
      prev_center = self.pos_to_center[d1['limb_to_pos'][limb]]
      new_center = self.pos_to_center[d2['limb_to_pos'][limb]]

      dist = np.linalg.norm(prev_center - new_center, ord = 2)
      cost += dist / self.costs['Distance normalizer']

    if time < self.costs['Time threshold']:
      '''
        Ex. normalizer = 300 ms, then
        400 ms since = 3/4 cost
        100 ms since = 3 cost
      '''
      time_factor = self.costs['Time normalizer'] / time
      cost *= time_factor
    if self.verbose: print(f'Move cost: {cost}')
    return cost


  def double_step_cost(self, d1: dict, d2: dict, time: float) -> float:
    '''
      Indirectly reward longer time since last foot movement. Cannot directly penalize by time since last foot movement in current graph representation

      Add cost for each limb that double steps
    '''
    dsa = self.doublestep_actions
    cost = 0
    for limb in d2['limb_to_pos']:
      if limb not in d1['limb_to_heel_action']:
        continue

      prev_heel = d1['limb_to_heel_action'][limb] in dsa
      prev_toe = d1['limb_to_toe_action'][limb] in dsa
      curr_heel = d2['limb_to_heel_action'][limb] in dsa
      curr_toe = d2['limb_to_toe_action'][limb] in dsa

      prev_step = prev_heel or prev_toe
      curr_step = curr_heel or curr_toe

      if prev_step and curr_step:
        cost += self.costs['Double step per limb']

    if time < self.costs['Time threshold']:
      '''
        Ex. normalizer = 300 ms, then
        400 ms since = 3/4 cost
        100 ms since = 3 cost
      '''
      time_factor = self.costs['Time normalizer'] / time
      cost *= time_factor

    if time >= self.costs['Time forgive double step']:
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
      if d['limb_to_heel_action'] in self.downpress:
        if d['limb_to_toe_action'] in self.downpress:
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


  '''
    Primary
  '''
  def get_cost(self, sa1: str, sa2: str, time: float = 1, verbose: bool = False) -> float:
    '''
    '''
    self.verbose = verbose

    d1 = self.parse_stanceaction(sa1)
    d2 = self.parse_stanceaction(sa2)
    cost = self.angle_cost(d2) + \
      self.foot_inversion_cost(d2) + \
      self.foot_pos_cost(d2) + \
      self.hold_change_cost(d1, d2) + \
      self.move_cost(d1, d2, time) + \
      self.double_step_cost(d1, d2, time) + \
      self.jump_cost(d1, d2) + \
      self.bracket_cost(d2) + \
      self.hands_cost(d2)
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