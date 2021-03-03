# 
from __future__ import division
import _config, _data, _stances, util, pickle, _params
import sys, os, fnmatch, datetime, subprocess, functools
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from more_itertools import unique_everseen

import _movement, _params, _memoizer

# Default params
inp_dir = _config.OUT_PLACE + f'b_graph/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

log_fn = ''

##
# Functions
##
def dijkstra(sc_nm, nodes, edges_out, edges_in, move_skillset = 'default'):
  '''
    nodes[node_nm] = {
      'Time': float,
      'Beat': float,
      'Line': str,
      'Line with active holds': str,
      'Measure': int,
      'BPM': float,
      'Stance actions': List[str],
      'Previous panels': List[str],
    }
    edges = {
      node_nm: List[node_nm: str]
    }
  
    Initial node has extra keys
    nodes['init']['Steptype'] = singles or doubles
    nodes['init']['Timing judge']
  '''
  steptype = nodes['init']['Steptype']
  mover = _movement.Movement(style=steptype, move_skillset=move_skillset)

  '''
    graph_nodes[node_nm][sa_idx] = (best_score,
                                    best_parent_node_nm,
                                    best_parent_sa_idx)
  '''
  graph_nodes = init_graph_nodes(nodes)
  stats_d = defaultdict(lambda: 0)

  @functools.lru_cache(maxsize=None)
  def get_parsed_stanceaction(sa):
    return mover.parse_stanceaction(sa)

  '''
    Investigate nodes
  '''
  # print('Exploring nodes in graph -- investigate for filtering and speedups')
  # nsas = []
  # for n in nodes:
  #   if n not in ['init', 'final']:
  #     nsa = len(nodes[n]['Stance actions'])
  #     nsas.append(nsa)
  # print(f'Num. stance actions per note line, statistics')
  # print(pd.DataFrame(nsas).describe())
  # n1 = edges_out['init'][0]
  # print(f'Stance actions of first note line')
  # print(nodes[n1]['Stance actions'])
  # import code; code.interact(local=dict(globals(), **locals()))

  print('Running Dijkstra`s algorithm ...')
  visited = set()
  node_qu = ['init']
  node_nms = list(nodes.keys())
  topo_sorted_nms = topological_sort(node_nms, edges_out)
  '''
    Traverse DAG with Dijkstra's in topological order
  '''
  timer = util.Timer(total = len(topo_sorted_nms))
  for nm in topo_sorted_nms:
    curr_sas = nodes[nm]['Stance actions']
    children = edges_out[nm]
    for child in children:
      child_sas = nodes[child]['Stance actions']
      timedelta = nodes[child]['Time'] - nodes[nm]['Time']
      child_line = nodes[child]['Line with active holds']
      is_multi = bool('multi' in child)

      if timedelta < 0.001:
        output_log('Notes are too close together, likely from very high bpm')
        sys.exit(1)

      for sa_idx, sa1 in enumerate(curr_sas):
        d1 = get_parsed_stanceaction(sa1)
        stance1 = sa1[:sa1.index(';')]

        consider_fast_jacks = False
        _, parent, parent_sa_idx = graph_nodes[nm][sa_idx]
        if parent is not None:
          prev_time = nodes[nm]['Time'] - nodes[parent]['Time']
          if prev_time < _params.jacks_footswitch_t_thresh and timedelta < _params.jacks_footswitch_t_thresh:
            sa_parent = nodes[parent]['Stance actions'][parent_sa_idx]
            d0 = get_parsed_stanceaction(sa_parent)

            mv_cost = _memoizer.move_cost(mover,
                sa_parent, stance1, d0, d1)
            if mv_cost <= 0:
              consider_fast_jacks = True

        '''
          Consider all children: filter by heuristics
          - Unnecessary jumps
          - Closest distance percentile
        '''
        subset_sa_idxs = []
        subset_sas = []
        # Todo -- consider a heap / priority queue
        subset_dists = []
        for sa_jdx, sa2 in enumerate(child_sas):
          d2 = get_parsed_stanceaction(sa2)
          stance2 = sa2[:sa2.index(';')]
          stats_d['Num. edges'] += 1

          # Do not filter edges going into final
          if child == 'final':
            subset_sa_idxs.append(sa_jdx)
            subset_sas.append(sa2)
            continue

          '''
            Get unnecessary jump flag by memoization
            Use strings as keys; dicts are not hashable

            TODO - Clean up this code; shorten?
            At beginning of for loops, d1 = get_parsed_stanceaction(stance1)
            -- Note: stance1 does not have action; makes cache more dense
            Make custom cache class?
          '''
          if nm != 'init':
            jump_flag = _memoizer.unnecessary_jump(mover,
                stance1, stance2, d1, d2, child_line)
            if jump_flag:
              stats_d['Num. edges skipped by unnecessary jump'] += 1
              continue

          '''
            Filter by skillset
          '''
          if move_skillset == 'beginner':
            beginner_flag = _memoizer.beginner_flag(mover, stance2, d2)
            if not beginner_flag:
              stats_d['Num. edges skipped by beginner filtering'] += 1
              continue

          subset_sa_idxs.append(sa_jdx)
          subset_sas.append(sa2)

        '''
          Filter subset sas by distance via sorting
        '''
        # pct = 99
        # dist_cutoff = np.percentile(subset_dists, pct)
        # dist_cutoff = min(subset_dists) * 10
        # prev_n = len(subset_sa_idxs)
        # subset_sa_idxs = [s for idx, s in enumerate(subset_sa_idxs) if subset_dists[idx] < dist_cutoff]
        # subset_sas = [s for idx, s in enumerate(subset_sas) if subset_dists[idx] < dist_cutoff]
        # stats_d['Num. edges skipped by dist'] += prev_n - len(subset_sa_idxs)

        '''
          Run Dijkstra's on subset of edges
        '''
        for sa_jdx, sa2 in zip(subset_sa_idxs, subset_sas):
          d2 = get_parsed_stanceaction(sa2)
          stance2 = sa2[:sa2.index(';')]

          stats_d['Num. edges considered'] += 1
          if child != 'final':
            # Get cost
            edge_cost = _memoizer.get_edge_cost(mover,
                sa1, sa2, d1, d2, timedelta, child)

            '''
              Modify cost w/o memoization
            '''
            # Fast jacks
            if consider_fast_jacks:
              mv_cost = _memoizer.move_cost(mover,
                  stance1, stance2, d1, d2)
              if mv_cost <= 0:
                edge_cost += mover.fast_jacks_cost(d0, d1, d2, prev_time, timedelta)

          elif child == 'final':
            edge_cost = 0

          # print(sa1, sa2, edge_cost)
          # import code; code.interact(local=dict(globals(), **locals()))

          curr_cost = graph_nodes[nm][sa_idx][0]
          cost = curr_cost + edge_cost

          if cost < graph_nodes[child][sa_jdx][0]:
            graph_nodes[child][sa_jdx] = (cost, nm, sa_idx)

    visited.add(nm)
    timer.update()

  # Save
  with open(out_dir + f'{sc_nm}.pkl', 'wb') as f:
    pickle.dump(graph_nodes, f)

  stats_d = _memoizer.add_cache_stats('psa', get_parsed_stanceaction, stats_d)
  stats_d = _memoizer.add_custom_memoizer_stats(stats_d)
  stats_df = pd.DataFrame(stats_d, index = ['Count']).T
  print(stats_df)

  # Find best path
  df = backtrack_annotate(graph_nodes, nodes)
  df.to_csv(out_dir + f'{sc_nm} {move_skillset}.csv')
  import code; code.interact(local=dict(globals(), **locals()))
  return


def backtrack_annotate(graph_nodes, nodes) -> pd.DataFrame:
  '''
    Backtrack from final node to init, getting stance actions
    Return df
  '''

  cols = [
    'Time',
    'Beat',
    'Line', 
    'Line with active holds',
    'Measure',
    'BPM',
  ]

  dd = defaultdict(list)
  cost, node, sa_idx = graph_nodes['final'][0]
  try:
    assert node is not None, f'Error in backtracking:   Final node has no parent'
  except AssertionError:
    print(f'Error in backtracking:   Final node has no parent')
    import code; code.interact(local=dict(globals(), **locals()))

  '''
    Backtrack
  '''
  while node != 'init':
    try:
      cost, parent_node, parent_sa_idx = graph_nodes[node][sa_idx]
    except KeyError:
      print(f'Error in backtracking: {node} has no parent')
      sys.exit(1)

    best_sa = nodes[node]['Stance actions'][sa_idx]
    ad = parse_sa_to_limb_action(best_sa)
    dd['Node name'].append(node)
    dd['Cost'].append(cost)
    dd['Stance action'].append(best_sa)
    for col in cols:
      dd[col].append(nodes[node][col])
    for limb in ad:
      dd[limb].append(ad[limb])

    node = parent_node
    sa_idx = parent_sa_idx

  df = pd.DataFrame(dd)
  df = df.iloc[::-1].reset_index(drop=True)
  df['Line'] = [f'`{s}' for s in df['Line']]
  df['Line with active holds'] = [f'`{s}' for s in df['Line with active holds']]

  '''
    Annotate in forward direction
  '''
  cdd = defaultdict(list)
  for idx, row in df.iterrows():
    if idx == 0:
      cdd['Jump'].append(np.nan)
    else:
      cdd['Jump'].append(is_jump(df.iloc[idx - 1]['Stance action'], row['Stance action']))
  for col in cdd:
    df[col] = cdd[col]
  return df

'''

'''
def is_jump(sa1, sa2):
  if sa1 == '':
    return np.nan
  stances1 = sa1[:sa1.index(';')].split(',')[:2]
  stances2 = sa2[:sa2.index(';')].split(',')[:2]
  jump_flag = True
  for s1, s2 in zip(stances1, stances2):
    if s1 == s2:
      jump_flag = False
  return 1 if jump_flag else np.nan



'''
  Helper
'''
def load_data(sc_nm: str):
  with open(inp_dir + f'{sc_nm}.pkl', 'rb') as f:
    nodes, edges_out, edges_in = pickle.load(f)
  return nodes, edges_out, edges_in


def init_graph_nodes(nodes: dict) -> dict:
  print(f'Initializing graph nodes ...')
  graph_nodes = {'init': {0: (0, None, None)}}
  timer = util.Timer(total=len(nodes))
  for node in nodes:
    if node == 'init':
      continue
    graph_nodes[node] = {}
    for sa_idx, sa in enumerate(nodes[node]['Stance actions']):
      graph_nodes[node][sa_idx] = (np.inf, None, None)
    timer.update()
  print('Done')
  return graph_nodes


def topological_sort(node_nms, edges_out):
  visited = [False] * len(node_nms)
  stack = []


  def topological_sort_util(idx, visited, stack):
    visited[idx] = True
    for child_nm in edges_out[node_nms[idx]]:
      jdx = node_nms.index(child_nm)
      if not visited[jdx]:
        topological_sort_util(jdx, visited, stack)
    stack.insert(0, node_nms[idx])
    return

  for idx in range(len(node_nms)):
    if not visited[idx]:
      topological_sort_util(idx, visited, stack)

  return stack


def parse_sa_to_limb_action(sa: str) -> dict:
  '''
    Tuple = (Left foot, right foot, left hand, right hand)
  '''
  actions = sa[sa.index(';'):].split(',') + ['--', '--']
  limbs = ('Left foot', 'Right foot', 'Left hand', 'Right hand')
  d = dict()
  res = []
  for limb, action in zip(limbs, actions):
    if '1' in action or '2' in action or '4' in action:
      d[limb] = 1
    else:
      d[limb] = np.nan
  return d


def get_graph_stats(nodes):
  dd = defaultdict(list)
  for node in nodes:
    if node in ['init', 'final']:
      continue
    dd['Num. stance actions'].append(len(nodes[node]['Stance actions']))
    dd['Node'].append(node)
  ndf = pd.DataFrame(dd)
  print('Num. stance actions per node, statistics:')
  print(ndf['Num. stance actions'].describe())
  # import code; code.interact(local=dict(globals(), **locals()))
  return


'''
  Logging
'''
def output_log(message):
  print(message)
  with open(log_fn, 'w') as f:
    f.write(message)
  return


##
# qsub
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  num_scripts = 0
  for idx in range(0, 10):
    command = f'python {NAME}.py {idx}'
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{idx}.sh'
    with open(sh_fn, 'w') as f:
      f.write(f'#!/bin/bash\n{command}\n')
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append(f'qsub -j y -V -wd {_config.SRC_DIR} {sh_fn}')

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print(f'Wrote {num_scripts} shell scripts to {qsubs_dir}')
  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  nm = 'Super Fantasy - SHK S19 arcade'
  # nm = 'Super Fantasy - SHK S7 arcade'
  # nm = 'Super Fantasy - SHK S4 arcade'
  # nm = 'Final Audition 2 - BanYa S7 arcade'
  # nm = 'Sorceress Elise - YAHPP S23 arcade'
  # nm = 'Super Fantasy - SHK S10 arcade'
  # nm = '1950 - SLAM S23 arcade'
  # nm = 'HTTP - Quree S21 arcade'
  # nm = '8 6 - DASU S20 arcade'
  # nm = 'Shub Sothoth - Nato & EXC S25 remix'
  # nm = 'The End of the World ft. Skizzo - MonstDeath S20 arcade'
  # nm = 'Loki - Lotze S21 arcade'
  # nm = 'Native - SHK S20 arcade'
  # nm = 'PARADOXX - NATO & SLAM S26 remix'
  # nm = 'BEMERA - YAHPP S24 remix'
  # nm = 'HEART RABBIT COASTER - nato S23 arcade'
  # nm = 'F(R)IEND - D_AAN S23 arcade'
  # nm = 'Pump me Amadeus - BanYa S11 arcade'
  # nm = 'King of Sales - Norazo S21 arcade'
  # nm = 'Hyperion - M2U S20 shortcut'
  # nm = 'Final Audition Ep. 2-2 - YAHPP S22 arcade'
  # nm = 'Achluoias - D_AAN S24 arcade'
  # nm = 'Awakening - typeMARS S16 arcade'

  # Doubles
  # nm = 'Mitotsudaira - ETIA. D19 arcade'
  # nm = 'Trashy Innocence - Last Note. D16 arcade'

  # move_skillset = 'beginner'
  move_skillset = 'basic'

  global log_fn
  log_fn = out_dir + f'{nm} {move_skillset}.log'
  util.exists_empty_fn(log_fn)
  print(nm, move_skillset)

  nodes, edges_out, edges_in = load_data(nm)
  get_graph_stats(nodes)
  dijkstra(nm, nodes, edges_out, edges_in, move_skillset=move_skillset)

  output_log('Success')
  return


if __name__ == '__main__':
  main()