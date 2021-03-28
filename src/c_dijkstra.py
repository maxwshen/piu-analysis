'''
  Dijkstra's algorithm on topologically sorted nodes: O(E + V).
  Faster big-O time than priority queue at O(E + VlogV), but
  priority queue allows skipping nodes, which could make it
  faster in practice.
'''
import _config, _data, _stances, util, _params
import sys, os, pickle, fnmatch, datetime, subprocess, functools
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from heapq import heappush, heappop

import _movement, _params, _memoizer
import _graph

# Default params
inp_dir = _config.OUT_PLACE + f'b_graph/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

log_fn = ''
stats_d = defaultdict(lambda: 0)

'''
  Primary
'''
def dijkstra(graph):
  # Updates graph.costs and graph.predecessor
  # TODO Re-implement Dynamic cost function for fast jacks
  global stats_d
  visited = set()
  qu = [(0, graph.init_node)]

  print('Running Dijkstra`s algorithm ...')
  timer = util.Timer()
  while qu:
    u_cost, u = heappop(qu)

    if u == graph.final_node:
      break
      
    if u in visited:
      continue

    visited.add(u)
    stats_d['Num. nodes considered'] += 1

    for v in graph.edge_generator(u):
      if v in visited or graph.filter_edge(u, v):
        continue
      
      graph.error_check(u, v)

      stats_d['Num. edges considered'] += 1
      v_cost = u_cost + graph.edge_cost(u, v)
      if v_cost < graph.costs[v]:
        graph.costs[v] = v_cost
        graph.predecessors[v] = u
        heappush(qu, (v_cost, v))
    # timer.update()

  # Record memoization stats
  stats_d = _memoizer.add_cache_stats('psa', graph.parse_sa, stats_d)
  stats_d = _memoizer.add_custom_memoizer_stats(stats_d)
  stats_d.update(graph.stats_d)
  stats_df = pd.DataFrame(stats_d, index = ['Count']).T
  print(stats_df)
  return graph


def backtrack_annotate(graph) -> pd.DataFrame:
  # Assumes graph.predecessors and graph.costs are populated by dijkstra
  cols = ['Time', 'Beat', 'Line', 'Line with active holds', 'Measure', 'BPM']
  dd = defaultdict(list)

  node = graph.predecessors[graph.final_node]
  cost = graph.costs[graph.final_node]

  if node is None:
    raise RunTimeError(f'Error in backtracking: Final node has no parent')

  while node != graph.init_node:
    parent = graph.predecessors[node]
    cost = graph.costs[node]
    if parent is None:
      raise RunTimeError(f'Error in backtracking: {node} has no parent')

    line, sa = _graph.parse_node_name(node)
    dd['Line node'].append(line)
    dd['Stance action'].append(sa)
    dd['Cost'].append(cost)
    for col in cols:
      dd[col].append(graph.line_nodes[line][col])
    ad = parse_sa_to_limb_action(sa)
    for limb in ad:
      dd[limb].append(ad[limb])

    node = parent

  df = pd.DataFrame(dd)
  df = df.iloc[::-1].reset_index(drop=True)
  excel_refmt = lambda s: f'`{s}'
  df['Line'] = [excel_refmt(s) for s in df['Line']]
  df['Line with active holds'] = [excel_refmt(s) for s in df['Line with active holds']]

  return df


'''
  Parsing
'''
@functools.lru_cache(maxsize=None)
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


'''
  IO
'''
def load_data(sc_nm: str):
  with open(inp_dir + f'{sc_nm}.pkl', 'rb') as f:
    line_nodes, line_edges_out, line_edges_in = pickle.load(f)
  return line_nodes, line_edges_out, line_edges_in


def output_log(message):
  print(message)
  with open(log_fn, 'w') as f:
    f.write(message)
  return


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


@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  # nm = 'Super Fantasy - SHK S19 arcade'
  # nm = 'Super Fantasy - SHK S7 arcade'
  nm = 'Super Fantasy - SHK S4 arcade'
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

  move_skillset = 'beginner'
  # move_skillset = 'basic'

  global log_fn
  log_fn = out_dir + f'{nm} {move_skillset}.log'
  util.exists_empty_fn(log_fn)
  print(nm, move_skillset)

  line_nodes, line_edges_out, line_edges_in = load_data(nm)

  steptype = line_nodes['init']['Steptype']
  mover = _movement.Movement(style=steptype, move_skillset=move_skillset)
  graph = _graph.Graph(mover, line_nodes, line_edges_out)

  graph.graph_stats()

  graph = dijkstra(graph)
  df = backtrack_annotate(graph)
  df.to_csv(out_dir + f'{nm} {move_skillset}.csv')

  output_log('Success')
  return


if __name__ == '__main__':
  main()