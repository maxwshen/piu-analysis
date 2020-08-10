#
import _data
import _config
from collections import defaultdict, Counter
import numpy as np, pandas as pd
import os, copy
from typing import List, Dict, Set, Tuple

singles_pos_df = pd.read_csv(_config.DATA_DIR + f'positions_singles.csv', index_col = 0)
doubles_pos_df = pd.read_csv(_config.DATA_DIR + f'positions_doubles.csv', index_col = 0)

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
      self.df = singles_pos_df
      self.idx_to_panel = {
        0: 'p1,1',
        1: 'p1,7',
        2: 'p1,5',
        3: 'p1,9',
        4: 'p1,3',
      }
    elif style == 'doubles':
      self.df = doubles_pos_df
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

    self.arrow_panels = list(self.idx_to_panel.values())
    self.all_limbs = ['Left foot', 'Right foot', 'Left hand', 'Right hand']

    self.limb_panel_to_footpos = self.__init_panel_to_footpos()

    self.panel_to_idx = {self.idx_to_panel[idx]: idx for idx in self.idx_to_panel}

    self.nm_to_heel_panel = {nm: p for nm, p in zip(self.df['Name'], self.df['Panel - heel'])}
    self.nm_to_toe_panel = {nm: p for nm, p in zip(self.df['Name'], self.df['Panel - toe'])}
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
  def get_stances(self, active_panels, prev_panels):
    '''
      - Determine how many limbs we need
      - Propose stances as combinations of allowed positions for each limb, subset by possibility (in csv) and panels
      - Filter to stances that cover constraint panels
    '''
    num_constraints = len(active_panels)
    all_panels = list(set(active_panels) | set(prev_panels))

    '''
      Todo -- smarter detection of whether we need hands or not. 
    '''
    limbs = ['Left foot', 'Right foot']
    if num_constraints > 4:
      limbs += ['Left hand', 'Right hand']

    ps = []
    for limb in limbs:
      positions = set().union(*[self.limb_panel_to_footpos[limb][panel] for panel in all_panels])
      ps.append(list(positions))

    sts = self.recursive_get_stances(ps)

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


  def recursive_get_stances(self, ps: List[List[str]]) -> List[List[str]]:
    '''
      List of <num_limbs> lists of positions
      Returns List of position combinations for each limb
    '''
    if len(ps) == 1:
      return [[s] for s in ps[0]]
    positions = self.recursive_get_stances(ps[:-1])
    np = []
    for p in positions:
      for s in ps[-1]:
        np.append(p + [s])
    return np


  '''
    Annotating actions
  '''
  def annotate_actions(self, panel_constraints: str, stances: List[str]) -> List[str]:
    '''
      Format: 
      Stance string ; Action string
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
    num_limbs = len(stance.split(','))
    action_template = []
    for idx in range(num_limbs):
      action_template.append(['-', '-'])
    extremity_to_jdx = {'heel': 0, 'toe': 1}

    ps = list(panel_to_part.keys())
    vs = list(panel_to_part.values())
    # print('vs', vs)
    part_combos = self.recursive_part_combos(vs)

    actions = []
    for part_combo in part_combos:
      action = copy.deepcopy(action_template)
      for panel, part in zip(ps, part_combo):
        if panel not in panel_to_action:
          continue
        constraint = panel_to_action[panel]
        [idx, extremity] = part
        jdx = extremity_to_jdx[extremity]
        action[idx][jdx] = constraint
      action = [''.join(s) for s in action]
      action_str = ','.join(action)
      actions.append(action_str)

    sas = [f'{stance};{action}' for action in actions]
    return sas


  def recursive_part_combos(self, parts: List[List]) -> List[List]:
    if len(parts) == 1:
      # print('zero', )
      return [[s] for s in parts[0]]
    combos = self.recursive_part_combos(parts[:-1])
    new_combos = []
    for combo in combos:
      for item in parts[-1]:
        new_combos.append(combo + [item])
    return new_combos


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

    # Annotate all possible actions (one to many relationship)
    stance_actions = self.annotate_actions(panel_constraints, stances)
    if verbose: print(stance_actions)

    return stance_actions


  '''
    Helper
  '''
  def text_to_panels(self, text: str) -> List[str]:
    '''
      '10002' -> ['p1,1', 'p1,3']
    '''
    return[self.idx_to_panel[idx] for idx, num in enumerate(text) if num != '0']


  def text_to_panel_to_action(self, text: str) -> dict:
    panel_to_action = dict()
    for idx, action in enumerate(text):
      panel = self.idx_to_panel[idx]
      if action != '0':
        panel_to_action[panel] = action
    return panel_to_action


  def initial_stanceaction(self):
    if self.style == 'singles':
      return ['14,36;--,--']
    if self.style == 'doubles':
      return []


  def combine_lines(self, lines: List[str]):
    '''
      '10000' and '00100' -> '10100'
    '''
    if len(lines) == 1:
      return lines[0]

    combined_line = ''
    n = len(lines[0])
    priority = ['4', '2', '1', '3', '0']

    for idx in range(n):
      cs = set([line[idx] for line in lines])
      for char in priority:
        if char in cs:
          combined_line += char
          break
    return combined_line


'''
  Testing
'''
def test():
  stance = Stances(style = 'singles')
  print(f'Running tests ...')

  # pattern = '10000'
  # pattern = '10001'
  pattern = '01110'

  print(f'Running {pattern} ...')
  stance_actions = stance.get_stanceactions(pattern, verbose = True)
  import code; code.interact(local=dict(globals(), **locals()))

  test_limb_order_preservation(stance)
  test_prev_panel_reduction(stance, verbose = True)
  test_continue_hold(stance)
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
  sa2 = stance.get_stanceactions(pattern, prev_panels = ['p1,1'])
  sa3 = stance.get_stanceactions(pattern, prev_panels = ['p1,1', 'p1,9'])
  sa4 = stance.get_stanceactions(pattern, prev_panels = ['p1,1', 'p1,3', 'p1,7', 'p1,9'])

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
  found_righthand_pos = set([s.split(';')[0].split(',')[-1] for s in stance_actions])
  df = stance.df
  expected_rh_pos = set(df[df['Right hand'] == True]['Name'])
  for idx, row in stance.df.iterrows():
    if row['Right hand']:
      assert row['Name'] in found_righthand_pos, f'{row["Name"]} not found'
    else:
      assert row['Name'] not in found_righthand_pos
  print('Passed')
  return


if __name__ == '__main__':
  test()