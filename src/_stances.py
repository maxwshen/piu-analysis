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

class SinglesStances():
  '''
    Intended usage: From an input set of constraints (panels to hit), return all stances that satisfy constraints as a set of nodes

    Also used for hands
  '''
  def __init__(self):
    self.df = singles_pos_df
    self.idx_to_panel = {
      0: 'p1,1',
      1: 'p1,7',
      2: 'p1,5',
      3: 'p1,9',
      4: 'p1,3',
      # 5: 'p2,1',
      # 6: 'p2,7',
      # 7: 'p2,5',
      # 8: 'p2,9',
      # 9: 'p2,3',
    }
    self.arrow_panels = list(self.idx_to_panel.values())
    self.all_limbs = ['Left foot', 'Right foot', 'Left hand', 'Right hand']

    self.limb_panel_to_footpos = self.__init_panel_to_footpos()

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
    Assigning limbs to panels
  '''
  def recursive_limb_combos(self, limbs: List, n: int) -> List[List]:
    if n == 1:
      return [[s] for s in limbs]
    rlas = self.recursive_limb_combos(limbs, n - 1)
    new_rlas = []
    for la in rlas:
      for limb in limbs:
        new_rlas.append(la + [limb])
    return new_rlas


  def check_valid(self, limb_assignment: List[str], active_panels: List[str]):
    limb_to_panels = defaultdict(list)
    for limb, panel in zip(limb_assignment, active_panels):
      limb_to_panels[limb].append(panel)

    for limb in limb_to_panels:
      panels = limb_to_panels[limb]
      footpos = set.intersection(*[self.limb_panel_to_footpos[limb][panel] for panel in panels])
      if len(footpos) == 0:
        return False, None

    # Enforce use of exactly 2 or 4 limbs
    if len(limb_to_panels.keys()) not in [2, 4]:
      return False, None

    # Enforce that dict is ordered by self.all_limbs
    l_to_p = {limb: limb_to_panels[limb] for limb in self.all_limbs if limb in limb_to_panels}
    return True, l_to_p


  def get_limb_assignments(self, panel_constraints: str) -> List[dict]:
    '''
    '''
    num_constraints = len(panel_constraints) - panel_constraints.count('0')
    active_panels = [self.idx_to_panel[idx] for idx, num in enumerate(panel_constraints) if num != '0']

    limbs = ['Left foot', 'Right foot']
    if num_constraints > 4:
      limbs += ['Left hand', 'Right hand']

    rlas = self.recursive_limb_combos(limbs, len(active_panels))

    # Filter limb assignments
    valid_las = []
    for limb_assignment in rlas:
      ok_flag, limb_to_panels = self.check_valid(limb_assignment, active_panels)
      if ok_flag:
        valid_las.append(limb_to_panels)

    # print(valid_las)
    return valid_las

  '''
    Annotating foot positions
  '''
  def expand_to_footpos(self, las: List[dict]) -> List[str]:
    all_stances = []
    for la in las:
      limb_to_footpos = dict()
      for limb in la:
        panels = la[limb]
        foot_pos = set.intersection(*[self.limb_panel_to_footpos[limb][panel] for panel in panels])
        limb_to_footpos[limb] = list(foot_pos)

      stances = self.get_stance_strs(limb_to_footpos)
      for stance in stances:
        if stance not in all_stances:
          all_stances.append(stance)
    return all_stances


  def get_stance_strs(self, limb_to_footpos: dict) -> List[str]:
    '''
      Creates a stance string, delimiter = ','
      limb_to_footpos assumed to follow order:
      - Left foot
      - Right foot
      - Left hand (optional)
      - Right hand (optional)
    '''
    delim = ','
    stance_strs = []
    for limb in limb_to_footpos:
      if len(stance_strs) == 0:
        stance_strs = limb_to_footpos[limb]
      else:
        new_ss = []
        for s in stance_strs:
          for footpos in limb_to_footpos[limb]:
            new_ss.append(s + f'{delim}{footpos}')
        stance_strs = new_ss
    return stance_strs


  '''
    Annotating actions
  '''
  def annotate_actions(self, panel_constraints: str, stances: List[str]) -> List[str]:
    '''
      Format: 
      Stance string ; Action string
    '''
    panel_to_action = dict()
    for idx, action in enumerate(panel_constraints):
      panel = self.idx_to_panel[idx]
      if action != '0':
        panel_to_action[panel] = action

    stance_actions = []
    for stance in stances:
      poss = stance.split(',')

      panel_to_part = defaultdict(list)
      for idx, pos in enumerate(poss):
        # design_row = self.df[self.df['Name'] == pos].iloc[0]
        # heel_panel = design_row['Panel - heel']
        # toe_panel = design_row['Panel - toe']
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
  def get_stanceactions(self, panel_constraints: str, verbose = False) -> List[str]:
    '''
      panel_constraints: '10002'

      stance_action: example 15,53;1-,-1
      <limb positions>;<limb actions>
      For each subgroup, comma-delimited position names for limbs in [Left foot, Right foot, Left hand, Right hand].

      stance_actions are unique and represent nodes in the graph.
    '''
    # Assign limbs to each panel
    las = self.get_limb_assignments(panel_constraints)
    if verbose: print(las)

    # Expand limb-panel tuples to all foot positions
    stances = self.expand_to_footpos(las)
    if verbose: print(stances)
    # import code; code.interact(local=dict(globals(), **locals()))

    # Annotate all possible actions
    stance_actions = self.annotate_actions(panel_constraints, stances)
    if verbose: print(stance_actions)

    return stance_actions



def test():
  stance = SinglesStances()
  print(f'Running tests ...')
  # pattern = '10001'
  # pattern = '01110'

  '''
    Test with '11111' pattern: Check that all righthand positions proposed are concordant with design
  '''

  pattern = '11111'
  print(f'Running {pattern} ...')
  print(f'... checking that proposed hand positions are concordant with design')
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