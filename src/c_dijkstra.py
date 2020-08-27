# 
from __future__ import division
import _config, _data, _stances, util, pickle, _params
import sys, os, fnmatch, datetime, subprocess
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from more_itertools import unique_everseen
from functools import lru_cache

import _movement, _params

# Default params
inp_dir = _config.OUT_PLACE + f'b_graph/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

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
  mover = _movement.Movement(style = steptype, move_skillset = move_skillset)

  '''
    graph_nodes[node_nm][sa_idx] = (best_score, best_parent_node_nm, best_parent_sa_idx)
  '''
  graph_nodes = init_graph_nodes(nodes)
  cost_memoizer = dict()
  jump_memoizer = dict()
  dist_memoizer = dict()
  psa_memoizer = dict()
  beginner_memoizer = dict()
  stats_d = defaultdict(lambda: 0)


  @lru_cache(maxsize = None)
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

      for sa_idx, sa1 in enumerate(curr_sas):
        d1 = get_parsed_stanceaction(sa1)
        stance1 = sa1[:sa1.index(';')]

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
          '''
          if nm != 'init':
            jump_key = (stance1, stance2, child_line)
            if jump_key in jump_memoizer:
              jump_flag = jump_memoizer[jump_key]
              stats_d['Jump memoizer, num hits'] += 1
            else:
              jump_flag = mover.unnecessary_jump(d1, d2, child_line)
              jump_memoizer[jump_key] = jump_flag
            if jump_flag:
              stats_d['Num. edges skipped by unnecessary jump'] += 1
              continue

          '''
            Filter by skillset
          '''
          if move_skillset == 'beginner':
            bg_key = stance2
            if bg_key in beginner_memoizer:
              beginner_flag = beginner_memoizer[bg_key]
              stats_d['Beginner memoizer, num hits'] += 1
            else:
              beginner_flag = mover.beginner_ok(d2)
              beginner_memoizer[bg_key] = beginner_flag
            if not beginner_flag:
              stats_d['Num. edges skipped by beginner filtering'] += 1
              continue

          '''
            Record distance for filtering later by sorting
          '''
          # dist_key = (stance1, stance2)
          # if dist_key in dist_memoizer:
          #   dist = dist_memoizer[dist_key]
          #   stats_d['Dist memoizer, num hits'] += 1
          # else:
          #   dist = mover.move_cost(d1, d2, time = 1)
          #   dist_memoizer[dist_key] = dist
          # subset_dists.append(dist)

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
            # Get cost by memoization
            if (sa1, sa2, timedelta) in cost_memoizer:
              edge_cost = cost_memoizer[(sa1, sa2, timedelta)]
              stats_d['Cost memoizer, num hits'] += 1

            else:
              # edge_cost = mover.get_cost(sa1, sa2, time = timedelta)
              edge_cost = mover.get_cost_from_ds(d1, d2, time = timedelta)

              # Multihit modifier if brackets
              multi_mod = mover.multihit_modifier(d1, d2, child)
              edge_cost += multi_mod

              # Apply time cost here to get memoization speedup and time sensitivity
              if 0.001 < timedelta < mover.costs['Time threshold']:
                time_factor = mover.costs['Time normalizer'] / timedelta
                edge_cost *= time_factor

              cost_memoizer[(sa1, sa2, timedelta)] = edge_cost

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

  cache_funcs = {
    'psa cache': get_parsed_stanceaction,
  }
  for cache_func in cache_funcs:
    info = cache_funcs[cache_func].cache_info()
    stats_d[f'{cache_func}, num hits'] = info[0]
    stats_d[f'{cache_func}, size'] = info[-1]
  stats_d['Cost memoizer, size'] = len(cost_memoizer)
  stats_d['Jump memoizer, size'] = len(jump_memoizer)

  stats_df = pd.DataFrame(stats_d, index = ['Count']).T
  print(stats_df)

  # Find best path
  df = backtrack_annotate(graph_nodes, nodes)
  df.to_csv(out_dir + f'{sc_nm}.csv')
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
  df = df.iloc[::-1].reset_index(drop = True)
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
  timer = util.Timer(total = len(nodes))
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
    command = 'python %s.py %s' % (NAME, idx)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, idx)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -j y -V -wd %s %s' % (_config.SRC_DIR, sh_fn))

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print('Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir))
  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  # nm = 'Super Fantasy - SHK S19 arcade'
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
  nm = 'PARADOXX - NATO & SLAM S26 remix'


  # move_skillset = 'beginner'
  move_skillset = 'basic'
  # move_skillset = 'advanced'

  print(nm, move_skillset)

  nodes, edges_out, edges_in = load_data(nm)
  dijkstra(nm, nodes, edges_out, edges_in, move_skillset = move_skillset)
  return


if __name__ == '__main__':
  main()