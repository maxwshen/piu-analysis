# 
from __future__ import division
import _config, _data, _stances, util, pickle, _params
import sys, os, fnmatch, datetime, subprocess
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from more_itertools import unique_everseen

import _movement, _params

# Default params
inp_dir = _config.OUT_PLACE + f'b_graph/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

##
# Functions
##
def dijkstra(sc_nm, nodes, edges_out, edges_in):
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
  mover = _movement.Movement(style = steptype)

  '''
    graph_nodes[node_nm][sa_idx] = (best_score, best_parent_node_nm, best_parent_sa_idx)
  '''
  graph_nodes = init_graph_nodes(nodes)
  cost_memoizer = dict()
  jump_memoizer = dict()
  psa_memoizer = dict()
  stats_d = defaultdict(lambda: 0)

  print('Running Dijkstra`s algorithm ...')
  visited = set()
  node_qu = ['init']
  timer = util.Timer(total = len(nodes))
  '''
    Traverse DAG with Dijkstra's
    todo: implement topological sort
  '''
  while len(node_qu) > 0:
    nm, node_qu = node_qu[0], node_qu[1:]
    children = edges_out[nm]
    node_qu += children

    curr_sas = nodes[nm]['Stance actions']

    for child in children:
      child_sas = nodes[child]['Stance actions']
      timedelta = nodes[child]['Time'] - nodes[nm]['Time']
      child_line = nodes[child]['Line with active holds']

      for sa_idx, sa1 in enumerate(curr_sas):
        d1 = get_parsed_stanceaction(sa1, psa_memoizer, stats_d, mover)
        stance1 = sa1[:sa1.index(';')]
        for sa_jdx, sa2 in enumerate(child_sas):
          d2 = get_parsed_stanceaction(sa2, psa_memoizer, stats_d, mover)
          stance2 = sa2[:sa2.index(';')]
          stats_d['Num. edges'] += 1

          # Get unnecessary jump flag by memoization
          jump_key = (stance1, stance2, child_line)
          if jump_key in jump_memoizer:
            jump_flag = jump_memoizer[jump_key]
            stats_d['Num. times jump memoizer used'] += 1
          else:
            jump_flag = mover.unnecessary_jump(d1, d2, child_line)
            jump_memoizer[jump_key] = jump_flag

          if jump_flag:
            stats_d['Num. edges skipped by unnecessary jump'] += 1
            continue

          if child != 'final':
            # Get cost by memoization
            if (sa1, sa2) in cost_memoizer:
              edge_cost = cost_memoizer[(sa1, sa2)]
              stats_d['Num. times cost memoizer used'] += 1

            else:
              # edge_cost = mover.get_cost(sa1, sa2, time = timedelta)
              edge_cost = mover.get_cost_from_ds(d1, d2)
              # Todo -- consider applying time cost here
              cost_memoizer[(sa1, sa2)] = edge_cost

          elif child == 'final':
            edge_cost = 0

          # print(sa1, sa2, edge_cost)

          curr_cost = graph_nodes[nm][sa_idx][0]
          cost = curr_cost + edge_cost

          if cost < graph_nodes[child][sa_jdx][0]:
            graph_nodes[child][sa_jdx] = (cost, nm, sa_idx)

    visited.add(nm)
    timer.update()

  # Save
  with open(out_dir + f'{sc_nm}.pkl', 'wb') as f:
    pickle.dump(graph_nodes, f)

  stats_df = pd.DataFrame(stats_d, index = ['Count']).T
  print(stats_df)

  # Find best path
  df = get_best_path(graph_nodes, nodes)
  df.to_csv(out_dir + f'{sc_nm}.csv')
  import code; code.interact(local=dict(globals(), **locals()))
  return


def get_best_path(graph_nodes, nodes):
  '''
    Backtrack from final node to init, getting stance actions
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
  while node != 'init':
    cost, parent_node, parent_sa_idx = graph_nodes[node][sa_idx]

    dd['Node name'].append(node)
    dd['Cost'].append(cost)
    dd['Stance action'].append(nodes[node]['Stance actions'][sa_idx])
    for col in cols:
      dd[col].append(nodes[node][col])

    node = parent_node
    sa_idx = parent_sa_idx

  df = pd.DataFrame(dd)
  df = df.iloc[::-1].reset_index(drop = True)
  df['Line'] = [f'`{s}' for s in df['Line']]
  df['Line with active holds'] = [f'`{s}' for s in df['Line with active holds']]
  return df


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


def get_parsed_stanceaction(sa, psa_memoizer, stats_d, mover):
  if sa in psa_memoizer:
    d = psa_memoizer[sa]
    stats_d['Num. times psa memoizer used'] += 1
  else:
    d = mover.parse_stanceaction(sa)
    psa_memoizer[sa] = d
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
  nm = 'Super Fantasy - SHK S19 arcade'
  # nm = 'Sorceress Elise - YAHPP S23 arcade'

  nodes, edges_out, edges_in = load_data(nm)
  dijkstra(nm, nodes, edges_out, edges_in)
  return


if __name__ == '__main__':
  main()