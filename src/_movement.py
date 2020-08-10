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
  def angle_cost(self, d: dict, verbose = False) -> float:
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
        cost += 3
      if angle < 0:
        cost += 0.5
      if angle < -45:
        cost += 3

    if verbose: print(f'Angle cost: {cost}')
    return cost


  def foot_inversion_cost(self, d: dict, verbose = False):
    '''
      Penalize if the right foot is left of left foot
    '''
    cost = 0
    left_coord = self.pos_to_center[d['limb_to_pos']['Left foot']]
    right_coord = self.pos_to_center[d['limb_to_pos']['Right foot']]

    if right_coord[0] < left_coord[0]:
      cost += self.costs['Inverted feet']

    if verbose: print(f'Foot inversion cost: {cost}')
    return cost


  def hold_change_cost(self, d1: dict, d2: dict, verbose = False):
    '''
      Penalize switching limb/part on a hold
    '''
    cost = 0

    if verbose: print(f'Hold change cost: {cost}')
    return cost


  def foot_pos_cost(self, d: dict, verbose = False) -> float:
    '''
      Sum over limbs
    '''
    cost = 0
    for limb in d['limb_to_pos']:
      pos = d['limb_to_pos'][limb]
      cost += self.pos_to_cost[pos]
    if verbose: print(f'Pos cost: {cost}')
    return cost


  def move_cost(self, d1: dict, d2: dict, verbose = False) -> float:
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

    if verbose: print(f'Move cost: {cost}')
    return cost


  def double_step_cost(self, d1: dict, d2: dict, verbose = False) -> float:
    '''
      Indirectly reward longer time since last foot movement. Cannot directly penalize by time since last foot movement in current graph representation

      Add cost for each limb that double steps
    '''
    cost = 0
    for limb in d2['limb_to_pos']:
      if limb not in d1['limb_to_heel_action']:
        continue
      prev_step = False
      curr_step = False
      if d1['limb_to_heel_action'][limb] != '-' or d1['limb_to_toe_action'][limb] != '-':
        prev_step = True

      if d2['limb_to_heel_action'][limb] != '-' or d2['limb_to_toe_action'][limb] != '-':
        curr_step = True

      if prev_step and curr_step:
        cost += self.costs['Double step per limb']
    if verbose: print(f'Double step cost: {cost}')
    return cost


  def bracket_cost(self, d: dict, verbose = False) -> float:
    '''
      Sum over limbs
    '''
    cost = 0
    for limb in d['limb_to_pos']:
      if d['limb_to_heel_action'] in self.downpress:
        if d['limb_to_toe_action'] in self.downpress:
          cost += self.costs['Bracket']
    if verbose: print(f'Bracket cost: {cost}')
    return cost


  def hands_cost(self, d: dict, verbose = False) -> float:
    '''
      Sum over limbs
    '''
    cost = 0
    if len(d['limb_to_pos']) > 2:
      cost += self.costs['Hands']
    if verbose: print(f'Hands cost: {cost}')
    return cost


  '''
    Primary
  '''
  def get_cost(self, sa1: str, sa2: str, verbose: bool = False) -> float:
    '''
    '''

    d1 = self.parse_stanceaction(sa1)
    d2 = self.parse_stanceaction(sa2)
    cost = self.angle_cost(d2, verbose = verbose) + \
      self.foot_inversion_cost(d2, verbose = verbose) + \
      self.foot_pos_cost(d2, verbose = verbose) + \
      self.hold_change_cost(d1, d2, verbose = verbose) + \
      self.move_cost(d1, d2, verbose = verbose) + \
      self.double_step_cost(d1, d2, verbose = verbose) + \
      self.bracket_cost(d2, verbose = verbose) + \
      self.hands_cost(d2, verbose = verbose)
    return cost


'''
  Testing
'''
def test():
  mover = Movement(style = 'singles')

  sa1 = '44,66;--,--'

  # Hitting 01000 = p1,7
  sa2s = [
    '47,36;-1,--',
    '57,36;-1,--',
    'a9,57;--,-1',
  ]

  for sa2 in sa2s:
    cost = mover.get_cost(sa1, sa2, verbose = True)
    print(sa2, cost, '\n')

  return


if __name__ == '__main__':
  test()