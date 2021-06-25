'''
  Dijkstra's algorithm on topologically sorted nodes: O(E + V).
  Faster big-O time than priority queue at O(E + VlogV), but
  priority queue allows skipping nodes, which makes it faster in practice.
'''
import _config, _data, _stances, util, _params
import sys, os, pickle, fnmatch, datetime, subprocess, functools
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from heapq import heappush, heappop

import _movement, _params, _memoizer, _stances, _qsub
import _graph, b_graph, segment, segment_edit

# Default params
inp_dir_b = _config.OUT_PLACE + 'b_graph/'
inp_dir_segment = _config.OUT_PLACE + 'segment/'
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
  global stats_d
  visited = set()
  qu = [(0, graph.init_node)]
  visited_line_nodes = set()

  print('Running Dijkstra`s algorithm ...')
  timer = util.Timer(total=len(graph.line_nodes), print_interval=5e4)
  while qu:
    u_cost, u = heappop(qu)

    if u == graph.final_node:
      break
      
    if u in visited:
      continue

    visited.add(u)
    stats_d['Num. nodes considered'] += 1

    for v in graph.edge_generator(u):
      if v in visited:
        continue

      if graph.filter_edge(u, v):
        continue
      
      graph.error_check(u, v)

      stats_d['Num. edges considered'] += 1
      cost_dict = graph.edge_cost(u, v)
      cost = round(sum(cost_dict.values()), 2)
      v_cost = u_cost + cost
      if v_cost < graph.costs[v]:
        graph.costs[v] = v_cost
        graph.cost_dicts[v] = cost_dict
        graph.predecessors[v] = u
        heappush(qu, (v_cost, v))

    line_node, sa, tag = _graph.parse_node_name(u)
    if line_node not in visited_line_nodes:
      visited_line_nodes.add(line_node)
      timer.update()
  timer.end()

  if u != graph.final_node:
    print('Error: Dijkstra terminated')
    import code; code.interact(local=dict(globals(), **locals()))
    raise Exception(f'Graph lacks path to final node. Traversed {len(visited)} nodes.')

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

  node = graph.predecessors[graph.final_node]
  if node is None:
    raise Exception(f'Error in backtracking: Final node has no parent')

  dd = defaultdict(list)
  while node != graph.init_node:
    parent = graph.predecessors[node]
    cost = graph.costs[node]
    cost_dict = graph.cost_dicts[node]
    if parent is None:
      raise Exception(f'Error in backtracking: {node} has no parent')

    line_node, sa, tag = _graph.parse_node_name(node)
    dd['Node'].append(node)
    dd['Line node'].append(line_node)
    dd['Stance action'].append(sa)
    dd['Tag'].append(tag)
    dd['Running cost'].append(cost)

    beat = graph.line_nodes[line_node]['Beat']
    annot = graph.annots.get(beat, '')
    motif = graph.beat_to_motif.get(beat, '')
    dd['Annotation'].append(annot)
    dd['Motif'].append(motif)

    for col in cols:
      dd[col].append(graph.line_nodes[line_node][col])
    ad = parse_sa_to_limb_action(sa)
    for limb in ad:
      dd[limb].append(ad[limb])
    dd['Cost'].append(round(sum(cost_dict.values()), 2))
    for col in cost_dict:
      dd[col].append(cost_dict[col])

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


def is_final_node(node):
  line_node, sa = _graph.parse_node_name(node)
  return line_node == 'final'


'''
  IO
'''
def output_log(message):
  print(message)
  with open(log_fn, 'w') as f:
    f.write(message)
  return


'''
  Run
'''
def run_single(nm):
  # move_skillset = 'beginner'
  move_skillset = 'basic'

  global log_fn
  log_fn = out_dir + f'{nm} {move_skillset}.log'
  util.exists_empty_fn(log_fn)
  print(nm, move_skillset)

  line_nodes, line_edges_out, line_edges_in = b_graph.load_data(inp_dir_b, nm)
  annots, motifs = segment.load_annotations(inp_dir_segment, nm)

  steptype = line_nodes['init']['Steptype']
  mover = _movement.Movement(style=steptype, move_skillset=move_skillset)
  graph = _graph.Graph(mover, line_nodes, line_edges_out, annots, motifs)

  graph = dijkstra(graph)
  df = backtrack_annotate(graph)
  df.to_csv(out_dir + f'{nm} {move_skillset}.csv')

  # graph.interactive_debug()
  # output_log('Success')
  return


@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  # nm = 'Super Fantasy - SHK S19 arcade'
  # nm = 'Last Rebirth - SHK S15 arcade'
  # nm = 'NoNoNo - Apink S14 arcade'
  # nm = 'Rage of Fire - MAX S16 arcade'
  # nm = 'Obelisque - ESTi x M2U S17 arcade'
  # nm = 'I Want U - MAX S19 arcade'
  nm = 'Forgotten Vampire - WyvernP S18 arcade'
  # nm = 'Conflict - Siromaru + Cranky S17 arcade'
  # nm = 'Tepris - Doin S17 arcade'
  # nm = 'Final Audition 2 - BanYa S7 arcade'
  # nm = 'Sorceress Elise - YAHPP S23 arcade'
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
  # nm = '8 6 - DASU D21 arcade'
  # nm = 'Bad End Night - HitoshizukuP x yama D18 arcade'
  run_single(nm)
  return


if __name__ == '__main__':
  if len(sys.argv) == 1:
    main()
  else:
    if sys.argv[1] == 'gen_qsubs':
      _qsub.gen_qsubs(NAME, sys.argv[2])
    elif sys.argv[1] == 'run_qsubs':
      _qsub.run_qsubs(
        chart_fnm = sys.argv[2],
        start = sys.argv[3],
        end = sys.argv[4],
        run_single = run_single,
      )
    elif sys.argv[1] == 'run_single':
      run_single(sys.argv[2])