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
    Intended usage: From an input set of constraints (panels to hit), return all stances that satisfy constraints as a set of nodes

    Also used for hands
  '''
  def __init__(self, style = 'singles'):
    self.style = style

    if style == 'singles':
      self.df = singles_pos_df
    elif style == 'doubles':
      self.df = doubles_pos_df

    self.costs = _params.costs

    self.all_limbs = ['Left foot', 'Right foot', 'Left hand', 'Right hand']
    pass

  '''
    Helper
  '''
  def parse_stanceaction(self, sa: str) -> dict:
    [stance, action] = sa.split(';')
    limb_to_pos = {limb: pos for limb, pos in zip(self.all_limbs, stance.split(','))}

    limb_to_heelaction = {}
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
  def stance_cost(self, d: dict) -> float:
    cost = 0
    return cost


  def foot_pos_cost(self, d: dict) -> float:
    cost = 0
    return cost


  def move_cost(self, d1: dict, d2: dict) -> float:
    cost = 0
    return cost


  def double_step_cost(self, d1: dict, d2: dict) -> float:
    '''
      Indirectly reward longer time since last foot movement. Cannot directly penalize by time since last foot movement in current graph representation

      Add cost for each limb that double steps
    '''
    cost = 0
    next_limbs = list(d2['limb_to_heel_action'].keys())
    for limb in next_limbs:
      if d1['limb_to_heel_action'][limb] != '-' or d1['limb_to_toe_action'][limb]:
        prev_step = True

      if d2['limb_to_heel_action'][limb] != '-' or d2['limb_to_toe_action'][limb]:
        curr_step = True

      if prev_step and curr_step:
        cost += self.costs['Double step per limb']

    return cost

  '''
    Primary
  '''
  def get_cost(self, sa1, sa2):
    '''
    '''

    d1 = self.parse_stanceaction(sa1)
    d2 = self.parse_stanceaction(sa2)
    cost = self.stance_cost(d2) + \
      self.foot_pos_cost(d2) + \
      self.move_cost(d1, d2) + \
      self.double_step_cost(d1, d2)
    return cost


'''
  Testing
'''
def test():
  mover = Movement(style = 'singles')
  return


if __name__ == '__main__':
  test()