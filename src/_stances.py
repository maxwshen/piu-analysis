#
import _data
import _config
from collections import defaultdict, Counter
import numpy as np, pandas as pd
import os, copy, itertools
from typing import List, Dict, Set, Tuple

import _positions

'''
  Foot positions
'''
class Stances():
  '''
    Intended usage: From an input set of constraints (panels to hit), return all stances that satisfy constraints as a set of nodes

    Also used for hands
  '''
  def __init__(self, style = 'singles'):
    self.style = style

    if style == 'singles':
      self.df = _positions.singles_pos_df
      self.idx_to_panel = {
        0: 'p1,1',
        1: 'p1,7',
        2: 'p1,5',
        3: 'p1,9',
        4: 'p1,3',
      }
    elif style == 'doubles':
      self.df = _positions.doubles_pos_df
      self.idx_to_panel = {
        0: 'p1,1',
        1: 'p1,7',
        2: 'p1,5',
        3: 'p1,9',
        4: 'p1,3',
        5: 'p2,1',
        6: 'p2,7',
        7: 'p2,5',
        8: 'p2,9',
        9: 'p2,3',
      }

    # No unusual positions
    self.df = self.df[self.df['Unusual'] != 1]

    self.arrow_panels = list(self.idx_to_panel.values())
    self.all_limbs = ['Left foot', 'Right foot', 'Left hand', 'Right hand']
    self.limb_panel_to_footpos = self.__init_panel_to_footpos()
    self.panel_to_idx = {self.idx_to_panel[idx]: idx for idx in self.idx_to_panel}
    self.nm_to_heel_panel = {nm: p for nm, p in zip(self.df['Name'],
                                                    self.df['Panel - heel'])}
    self.nm_to_toe_panel = {nm: p for nm, p in zip(self.df['Name'],
                                                   self.df['Panel - toe'])}
    pass


  def __init_panel_to_footpos(self) -> dict:
    '''
      md[limb][panel] = set of foot positions
    '''
    md = dict()
    for limb in self.all_limbs:
      d = defaultdict(set)
      dfs = self.df[self.df[limb] == True]
      for idx, row in dfs.iterrows():
        nm = row['Name']
        for panel in self.arrow_panels:
          if row[panel]:
            d[panel].add(nm)
      md[limb] = d
    return md


  '''
    Get foot positions from active and previous panels
  '''
  def get_stances(self, active_panels, prev_panels, use_hands = False):
    '''
      - Determine how many limbs we need
      - Propose stances as combinations of allowed positions for each limb, subset by possibility (in csv) and panels
      - Filter to stances that cover constraint panels
    '''
    num_constraints = len(active_panels)
    all_panels = list(set(active_panels) | set(prev_panels))

    '''
      TODO -- smarter detection of whether we need hands or not. 
    '''
    limbs = ['Left foot', 'Right foot']
    if use_hands:
      limbs += ['Left hand', 'Right hand']

    ps = []
    for limb in limbs:
      positions = set().union(*[self.limb_panel_to_footpos[limb][panel] for panel in all_panels])
      ps.append(list(positions))

    sts = itertools.product(*ps)

    # Filter stances that do not include all active panels
    filt_sts = []
    for st in sts:
      covered_panels = set()
      for pos in st:
        covered_panels.add(self.nm_to_toe_panel[pos])
        covered_panels.add(self.nm_to_heel_panel[pos])
      ok = True
      for ap in active_panels:
        if ap not in covered_panels:
          ok = False
          break
      if ok:
        filt_sts.append(st)

    delim = ','
    stance_strs = [delim.join(s) for s in filt_sts]
    return stance_strs


  '''
    Annotating actions
  '''
  def annotate_actions(self, panel_constraints: str, stances: List[str]) -> List[str]:
    '''
      Annotate actions on top of stances that are consistent with panel constraints. 
      Returns list of stance-actions that hit all panels in panel constraints.

      A stance-action is a string f'{stance_str};{action_str}'.
    '''
    panel_to_action = self.text_to_panel_to_action(panel_constraints)
    stance_actions = []
    for stance in stances:
      poss = stance.split(',')

      panel_to_part = defaultdict(list)
      for idx, pos in enumerate(poss):
        heel_panel = self.nm_to_heel_panel[pos]
        toe_panel = self.nm_to_toe_panel[pos]
        if heel_panel in self.arrow_panels:
          panel_to_part[heel_panel].append((idx, 'heel'))
        if toe_panel in self.arrow_panels:
          panel_to_part[toe_panel].append((idx, 'toe'))

      sas = self.get_sas(stance, panel_to_action, panel_to_part)
      stance_actions += sas

    return stance_actions


  def get_sas(self, stance, panel_to_action, panel_to_part) -> List[str]:
    '''
      Get stance-actions
    '''
    num_limbs = len(stance.split(','))
    action_template = []
    for idx in range(num_limbs):
      action_template.append(['-', '-'])
    extremity_to_jdx = {'heel': 0, 'toe': 1}

    ps = list(panel_to_part.keys())
    vs = list(panel_to_part.values())
    # print('vs', vs)
    part_combos = itertools.product(*vs)

    actions = []
    for part_combo in part_combos:
      action = copy.deepcopy(action_template)
      for panel, part in zip(ps, part_combo):
        if panel in panel_to_action:
          constraint = panel_to_action[panel]
          [idx, extremity] = part
          jdx = extremity_to_jdx[extremity]
          action[idx][jdx] = constraint
      action = [''.join(s) for s in action]
      action_str = ','.join(action)
      actions.append(action_str)

    sas = [f'{stance};{action}' for action in actions]
    return sas


  '''
    Primary
  '''
  def get_stanceactions(self, panel_constraints: str, prev_panels: List[str] = [], verbose: bool = False) -> List[str]:
    '''
      stance_action: example 15,53;1-,-1
      <limb positions>;<limb actions>
      For each subgroup, comma-delimited position names for limbs in [Left foot, Right foot, Left hand, Right hand].

      stance_actions are unique and represent nodes in the graph.

      Any non-zero value is treated as an action. We use:
      0: nothing
      1: hit
      2: start hold
      3: end hold
      4: continue hold
    '''
    active_panels = self.text_to_panels(panel_constraints)
    if len(prev_panels) == 0:
      prev_panels = self.arrow_panels
    prev_panels = list(set(prev_panels))

    # Get foot stances consistent with active panels, and including previous panels
    stances = self.get_stances(active_panels, prev_panels)
    stances = list(set(stances))
    if len(stances) == 0:
      stances = self.get_stances(active_panels, prev_panels, use_hands = True)
      stances = list(set(stances))

    # Annotate all possible actions (one to many relationship)
    stance_actions = self.annotate_actions(panel_constraints, stances)
    if verbose: print(stance_actions)

    return list(set(stance_actions))


  '''
    Helper
  '''
  def text_to_panels(self, text: str) -> List[str]:
    '''
      Returns a list of panel names that are pressed in the input text string.
      e.g., '10002' -> ['p1,1', 'p1,3']
    '''
    return [self.idx_to_panel[idx] for idx, num in enumerate(text) if num != '0']


  def text_to_panel_to_action(self, text: str) -> dict:
    panel_to_action = dict()
    for idx, action in enumerate(text):
      panel = self.idx_to_panel[idx]
      if action != '0':
        panel_to_action[panel] = action
    return panel_to_action


  def initial_stanceaction(self):
    '''
      Defines the initial stance-action at the beginning of a chart.
      TODO - Consider moving to _params.py ?
    '''
    if self.style == 'singles':
      return ['14,36;--,--']
    if self.style == 'doubles':
      return ['p1`36c,p2`14c;--,--']


  def combine_lines(self, lines: List[str]) -> str:
    '''
      Combines a list of panel constraints, prioritizing 0 < 3 < 4 < 1 < 2.
      Used for finding a single set of panel constraints to hit to satisfy
      multiple notes within the timing window ("multihits").
      e.g., '10000' and '00100' -> '10100'
    '''
    order_lowtohigh = '03412'
    priority = lambda x: order_lowtohigh.index(x)
    n_pads = len(lines[0])
    max_actions = [max([l[i] for l in lines], key=priority)
                   for i in range(n_pads)]
    return ''.join(max_actions)


'''
  Testing
'''
def test():
  stance = Stances(style='singles')
  print(f'Running tests ...')

  # pattern = '10000'
  # pattern = '10001'
  pattern = '01110'

  print(f'Running {pattern} ...')
  stance_actions = stance.get_stanceactions(pattern, verbose=True)
  import code; code.interact(local=dict(globals(), **locals()))

  test_limb_order_preservation(stance)
  test_prev_panel_reduction(stance, verbose=True)
  test_continue_hold(stance)
  test_combine_lines(stance)
  return


def test_continue_hold(stance):
  pattern = '10004'
  print(f'Running {pattern} ...')
  print(f'... checking that hold continue works')
  sa = stance.get_stanceactions(pattern)
  # Manually inspect if needed
  # import code; code.interact(local=dict(globals(), **locals()))
  print('Passed')
  return


def test_prev_panel_reduction(stance, verbose = False):
  '''
    Test with '11111' pattern: Check that all righthand positions proposed are concordant with design
  '''
  pattern = '00100'
  print(f'Running {pattern} ...')
  print(f'... checking that specifying prev panels reduces possible stances')
  sa1 = stance.get_stanceactions(pattern)
  sa2 = stance.get_stanceactions(pattern, prev_panels=['p1,1'])
  sa3 = stance.get_stanceactions(pattern, prev_panels=['p1,1', 'p1,9'])
  sa4 = stance.get_stanceactions(pattern, prev_panels=['p1,1', 'p1,3', 'p1,7', 'p1,9'])

  assert len(sa2) < len(sa1), 'Failed'
  assert len(sa2) < len(sa3), 'Failed'
  assert len(sa3) < len(sa1), 'Failed'
  assert len(sa4) == len(sa1), 'Failed'

  if verbose:
    print(len(sa2), len(sa3), len(sa4), len(sa1))

  print('Passed')
  return


def test_limb_order_preservation(stance):
  '''
    Test with '11111' pattern: Check that all righthand positions proposed are concordant with design
  '''
  pattern = '11111'
  print(f'Running {pattern} ...')
  print(f'... checking that limb order is correct')
  stance_actions = stance.get_stanceactions(pattern, verbose = False)
  # TODO - Handle parsing elsewhere?
  found_righthand_pos = set([s.split(';')[0].split(',')[-1] for s in stance_actions])
  df = stance.df
  expected_rh_pos = set(df[df['Right hand'] == True]['Name'])
  assert found_righthand_pos.issubset(expected_rh_pos), 'Proposed an illegal right-hand position'
  print('Passed')
  return


def test_combine_lines(stance):
  print(f'... checking combine_lines')
  assert stance.combine_lines(['10000', '00101']) == '10101'
  assert stance.combine_lines(['10203', '00101']) == '10201'
  print('Passed')
  return

if __name__ == '__main__':
  test()