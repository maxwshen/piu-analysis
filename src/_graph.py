import functools
import numpy as np, pandas as pd
from collections import defaultdict

import _memoizer, _params

class Graph():
  def __init__(self, mover, line_nodes, line_edges):
    self.stats_d = defaultdict(lambda: 0)
    self.mover = mover    

    @functools.lru_cache(maxsize=None)
    def __parse_sa(sa):
      return mover.parse_stanceaction(sa)
    self.parse_sa = __parse_sa

    self.line_nodes = self.filter_nodes(line_nodes)
    self.line_edges = line_edges

    self.predecessors = defaultdict(lambda: None)
    self.costs = defaultdict(lambda: np.inf)

    self.init_node = next(self.node_generator('init'))
    self.final_node = next(self.node_generator('final'))
    self.costs[self.init_node] = 0

    self.debug()
    pass


  '''
    Graph functions
  '''
  def node_generator(self, line):
    for sa in self.line_nodes[line]['Stance actions']:
      yield get_node_name(line, sa)


  def edge_generator(self, node):
    line, sa = parse_node_name(node)
    for line2 in self.line_edges[line]:
      for node in self.node_generator(line2):
        yield node
    pass


  def edge_cost(self, node1, node2):
    line1, sa1 = parse_node_name(node1)
    line2, sa2 = parse_node_name(node2)
    timedelta = self.timedelta(node1, node2)
    d1 = self.parse_sa(sa1)
    d2 = self.parse_sa(sa2)
    if line2 == 'final':
      return 0
    else:
      return _memoizer.get_edge_cost(self.mover, sa1, sa2, d1, d2, timedelta, line2)


  def error_check(self, node1, node2):
    line1, sa1 = parse_node_name(node1)
    line2, sa2 = parse_node_name(node2)
    timedelta = self.timedelta(node1, node2)
    
    if timedelta < 0.001:
      output_log('Notes are too close together, likely from very high bpm')
      sys.exit(1)
    return


  '''
    Filtering
  '''
  def filter_nodes(self, line_nodes):
    for line in line_nodes:
      if line not in ['init', 'final']:
        line_nodes[line]['Stance actions'] = [sa
            for sa in line_nodes[line]['Stance actions']
            if not self.filter_node(line, sa)]
    return line_nodes


  def filter_node(self, line, sa):
    filters = [
      self.filter_node_beginner,
    ]
    return any([f(line, sa) for f in filters])


  def filter_node_beginner(self, line, sa):
    remove = True
    d = self.parse_sa(sa)
    if self.mover.move_skillset == 'beginner':
      remove = not _memoizer.beginner_ok(self.mover, get_stance_from_sa(sa), d)
      if remove:
        self.stats_d['Num. nodes skipped by beginner filtering'] += 1
    return remove


  def filter_edge(self, node1, node2):
    # Ignore edges with heuristics
    filters = [
      self.filter_edge_unnecessary_jump,
    ]
    return any([f(node1, node2) for f in filters])


  def filter_edge_unnecessary_jump(self, node1, node2):
    '''
      Get unnecessary jump flag by memoization
      Use strings as keys; dicts are not hashable
    '''
    line1, sa1 = parse_node_name(node1)
    line2, sa2 = parse_node_name(node2)
    stance1 = get_stance_from_sa(sa1)
    stance2 = get_stance_from_sa(sa2)
    d1 = self.parse_sa(sa1)
    d2 = self.parse_sa(sa2)
    remove = _memoizer.unnecessary_jump(self.mover, stance1, stance2, d1, d2, line2)
    if remove:
      self.stats_d['Num. edges skipped by unnecessary jump'] += 1
    return remove


  '''
    Helper
  '''
  @functools.lru_cache(maxsize=None)
  def timedelta(self, node1, node2):
    line1, sa1 = parse_node_name(node1)
    line2, sa2 = parse_node_name(node2)
    return self.line_nodes[line2]['Time'] - self.line_nodes[line1]['Time']
    

  def graph_stats(self):
    dd = defaultdict(list)
    for line in self.line_nodes:
      if line not in ['init', 'final']:
        dd['Num. stance actions'].append(len(self.line_nodes[line]['Stance actions']))
        dd['Line'].append(line)
    ndf = pd.DataFrame(dd)
    print('Num. stance actions per line, statistics:')
    print(ndf['Num. stance actions'].describe())
    return


  def debug(self):
    # Explore nodes in graph -- investigate for filtering and speedups
    self.graph_stats()
    n1 = self.line_edges['init'][0]
    print(f'Stance actions of first note line')
    print(self.line_nodes[n1]['Stance actions'])
    import code; code.interact(local=dict(globals(), **locals()))
    return


'''
  Parsing
'''
@functools.lru_cache(maxsize=None)
def get_node_name(line_nm: str, sa: str) -> str:
  return f'{line_nm}:{sa}'


@functools.lru_cache(maxsize=None)
def parse_node_name(node: str):
  [line_nm, sa] = node.split(':')
  return line_nm, sa


@functools.lru_cache(maxsize=None)
def get_stance_from_sa(sa: str) -> str:
  return sa[:sa.index(';')]

