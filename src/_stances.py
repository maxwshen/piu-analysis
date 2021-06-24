#
import _data
import _config
from collections import defaultdict, Counter
import numpy as np, pandas as pd
import os, copy, itertools, functools
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
    self.bracket_pos = self.__init_bracket_pos()
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


  def __init_bracket_pos(self) -> set:
    crit = (self.df[self.arrow_panels].apply(sum, axis='columns') == 2)
    return set(self.df[crit]['Name'])


  '''
    Get foot positions from active panels
  '''
  def get_stances(self, active_panels, prev_panels, use_brackets = False, use_hands = False):
    '''
      All stances covering active_panels, allowing limbs to rest on prev_panels
    '''
    all_panels = list(set(active_panels) | set(prev_panels))

    limbs = ['Left foot', 'Right foot']
    if use_hands:
      limbs += ['Left hand', 'Right hand']

    def get_positions(limb, panels):
      footposs = [self.get_footpos(limb, panel, use_brackets) for panel in panels]
      return list(set().union(*footposs))

    ps = [get_positions(limb, all_panels) for limb in limbs]
    stance_strs = [','.join(limbpos) for limbpos in itertools.product(*ps)]
    stance_strs = [stance for stance in stance_strs 
                   if self.covers_panels(stance, active_panels)]
    return list(set(stance_strs))


  '''
    Annotating actions
  '''
  def annotate_actions(self, aug_line: str, stances: List[str]) -> List[str]:
    '''
      Annotate actions on top of stances that are consistent with panel constraints. 
      Returns list of stance-actions that hit all panels in panel constraints.

      A stance-action is a string f'{stance_str};{action_str}'.
    '''
    stance_actions = []
    for stance in stances:
      stance_actions += self.annotate_action(aug_line, stance)
    return stance_actions


  @functools.lru_cache(maxsize=None)
  def annotate_action(self, aug_line, stance) -> List[str]:
    panel_to_action = self.line_to_panel_to_action(aug_line)
    poss = self.stance_to_limbposs(stance)
    panel_to_part = defaultdict(list)
    for idx, pos in enumerate(poss):
      heel_panel = self.nm_to_heel_panel[pos]
      toe_panel = self.nm_to_toe_panel[pos]
      if heel_panel in self.arrow_panels:
        panel_to_part[heel_panel].append((idx, 'heel'))
      if toe_panel in self.arrow_panels:
        panel_to_part[toe_panel].append((idx, 'toe'))
    return self.get_sas(stance, panel_to_action, panel_to_part)


  def get_sas(self, stance, panel_to_action, panel_to_part) -> List[str]:
    '''
      Get stance-actions
    '''
    num_limbs = len(self.stance_to_limbposs(stance))
    action_template = [['-', '-'] for limb in range(num_limbs)]
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

    return [f'{stance};{action}' for action in actions]


  '''
    Primary
  '''
  def get_stanceactions(self, aug_line: str,
        prev_panels: Set[str] = set(), verbose = False) -> List[str]:
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
    active_panels = self.line_to_active_panels(aug_line)
    if len(prev_panels) == 0:
      prev_panels = set(self.arrow_panels)

    # Get foot stances consistent with active panels, and including previous panels
    # Propose brackets or hands only if necessary based on num. active panels
    if len(active_panels) > 4:
      stances = self.get_stances(active_panels, prev_panels,
          use_brackets=True, use_hands=True)
    elif len(active_panels) >= 2:
      '''
        Note: Much slower (10x?) using 2-4, rather than 3-4 inclusive.
        However, sometimes we need to bracket 2 notes ...
      '''
      stances = self.get_stances(active_panels, prev_panels,
          use_brackets=True)
    else:
      stances = self.get_stances(active_panels, prev_panels)

    # If heuristic failed
    if len(stances) == 0:
      stances = self.get_stances(active_panels, prev_panels,
          use_brackets=True, use_hands=True)

    # Annotate all possible actions (one to many relationship)
    stance_actions = self.annotate_actions(aug_line, stances)
    if verbose: print(stance_actions)
    return stance_actions


  @functools.lru_cache(maxsize=None)
  def stanceaction_generator(self, stance: str, aug_line: str):
    '''
      Input: Current stance, next line w/ holds
      Generates stance-actions that satisfy next line and move the min possible number of limbs from stance.

      Note: Checking for one move stances is empirically slower than just getting all stanceactions. Looks like unnecessary jump edge filtering is faster than one move stances. (3x slower for Loki s21)
    '''
    prev_feet_panels = self.stance_to_covered_panels(stance, feet_only=True)
    # active_panels = self.line_to_active_panels(aug_line)
    # one_move_stances, min_pads_per_foot = self.stances_by_moving_one_foot(
    #     stance, aug_line)
    # # if len(one_move_stances) > 0 and min_pads_per_foot == 1:
    # if len(one_move_stances) > 0:
    #   stance_actions = self.annotate_actions(aug_line, one_move_stances)
    # else:
    #   stance_actions = self.get_stanceactions(aug_line, prev_feet_panels)
    stance_actions = self.get_stanceactions(aug_line, prev_feet_panels)
    return stance_actions


  '''
    Helper
  '''
  def combine_lines(self, lines: List[str]) -> str:
    '''
      Combines a list of panel constraints, prioritizing 0 < 3 < 4 < 1 < 2.
      Used for finding a single set of panel constraints to hit to satisfy
      multiple notes within the timing window ("multihits").
      e.g., '10000' and '00100' -> '10100'
    '''
    priority = lambda x: '03412'.index(x)
    n_pads = len(lines[0])
    max_actions = [max([l[i] for l in lines], key=priority)
                   for i in range(n_pads)]
    return ''.join(max_actions)


  @functools.lru_cache(maxsize=None)
  def stances_by_moving_one_foot(self, stance: str, aug_line: str):
    '''
      Deprecated: Slower than proposing all stanceactions and filtering later by unnecessary jump
    '''
    active_panels = self.line_to_active_panels(aug_line)
    limb_to_panel = self.limb_to_panel_from_stance(stance)
    limb_to_pos = self.limb_to_pos_from_stance(stance)
    stances = []
    min_pads_per_foot = np.inf
    for stay_limb, move_limb in (('Left foot', 'Right foot'),
                                 ('Right foot', 'Left foot')):
      panels = [p for p in active_panels if p not in limb_to_panel[stay_limb]]
      min_pads_per_foot = min(len(panels), min_pads_per_foot)
      # Allow moving limb to panels covered by other limb for fast jacks
      if len(panels) == 0 and len(active_panels) <= 2:
        panels = active_panels
      footposs = set.intersection(*[self.limb_panel_to_footpos[move_limb][p]
                                    for p in panels])

      stances += self.build_stances({stay_limb: [limb_to_pos[stay_limb]],
                                     move_limb: list(footposs)})
    return stances, min_pads_per_foot


  def covers_panels(self, stance: str, active_panels: List[str]):
    # Does stance cover all active panels?
    covered_panels = self.stance_to_covered_panels(stance)
    return all([ap in covered_panels for ap in active_panels])


  def build_stances(self, limb_d):
    # limb_d[limb] = list of foot positions
    sorted_limbs = [limb for limb in self.all_limbs if limb in limb_d]
    poslists = [limb_d[limb] for limb in sorted_limbs]
    return [','.join(combo) for combo in itertools.product(*poslists)]


  @functools.lru_cache(maxsize=None)
  def get_footpos(self, limb, panel, use_brackets):
    fps = self.limb_panel_to_footpos[limb][panel]
    if not use_brackets:
      fps = [fp for fp in fps if fp not in self.bracket_pos]
    return fps


  '''
    Parsing
  '''
  @functools.lru_cache(maxsize=None)
  def line_to_active_panels(self, line: str) -> List[str]:
    '''
      Returns a list of panel names that are pressed in the input text string.
      e.g., '10002' -> ['p1,1', 'p1,3']
    '''
    return [self.idx_to_panel[idx] for idx, num in enumerate(line) if num != '0']


  @functools.lru_cache(maxsize=None)
  def line_to_panel_to_action(self, line: str) -> dict:
    panel_to_action = dict()
    for idx, action in enumerate(line):
      panel = self.idx_to_panel[idx]
      if action != '0':
        panel_to_action[panel] = action
    return panel_to_action


  @functools.lru_cache(maxsize=None)
  def limb_to_panel_from_stance(self, stance: str):
    return {limb: set([self.nm_to_toe_panel[pos], self.nm_to_heel_panel[pos]])
            for limb, pos in self.limb_to_pos_from_stance(stance).items()}


  @functools.lru_cache(maxsize=None)
  def stance_to_covered_panels(self, stance: str, feet_only = False):
    poss = self.stance_to_limbposs(stance, feet_only=feet_only)
    return set(self.nm_to_toe_panel[pos] for pos in poss) | \
           set(self.nm_to_heel_panel[pos] for pos in poss)


  @functools.lru_cache(maxsize=None)
  def stance_to_limbposs(self, stance: str, feet_only = False):
    if feet_only:
      return stance.split(',')[:2]
    else:
      return stance.split(',')


  @functools.lru_cache(maxsize=None)
  def limb_to_pos_from_stance(self, stance: str):
    return {limb: pos for limb, pos in
            zip(self.all_limbs, self.stance_to_limbposs(stance))}


  def limbs_downpress(self, sa):
    '''
      Get limb indices with downpresses in stanceaction
    '''
    actions = sa[sa.index(';'):]
    limbs = actions.split(',')
    has_action = lambda limb_action: any(x in limb_action for x in list('123'))
    return [i for i, la in enumerate(limbs) if has_action(la)]


  def limbs_doing(self, sa, doing):
    # doing = list('12')
    actions = sa[sa.index(';'):]
    limbs = actions.split(',')
    has_action = lambda limb_action: any(x in limb_action for x in doing)
    return [i for i, la in enumerate(limbs) if has_action(la)]

'''
  Testing
'''
def test():
  stance = Stances(style='singles')
  print(f'Running tests ...')

  # pattern = '10000'
  # pattern = '10001'
  # pattern = '01110'
  pattern = '11001'

  print(f'Running {pattern} ...')
  stance_actions = stance.get_stanceactions(pattern, verbose=True)

  test_limb_order_preservation(stance)
  test_continue_hold(stance)
  test_combine_lines(stance)
  test_one_move_stances(stance)
  return


def test_one_move_stances(stance):
  print(f'Checking proposing stances by moving one foot')
  stances = stance.stances_by_moving_one_foot('14,36', '10000')
  assert '14,36' in stances

  stances = stance.stances_by_moving_one_foot('14,36', '00100')
  assert '14,36' not in stances

  stances = stance.stances_by_moving_one_foot('14,36', '01010')
  assert len(stances) == 0

  stances = stance.stances_by_moving_one_foot('14,36', '00101')
  assert len(stances) > 0

  print('Passed')
  # Manually inspect if needed
  # import code; code.interact(local=dict(globals(), **locals()))
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