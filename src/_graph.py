'''
  Graph for Dijkstra's.
  Nodes are stance-actions at specific lines.
'''
import functools
import numpy as np, pandas as pd
from collections import defaultdict
from typing import List, Dict, Set, Tuple

import _memoizer, _params, _stances

class Graph():
  def __init__(self, mover, line_nodes, line_edges):
    self.stats_d = defaultdict(lambda: 0)
    self.mover = mover    
    self.stances = _stances.Stances(style=mover.style)

    @functools.lru_cache(maxsize=None)
    def __parse_sa(sa):
      return mover.parse_stanceaction(sa)
    self.parse_sa = __parse_sa

    self.line_nodes = line_nodes
    self.line_edges = line_edges

    self.predecessors = defaultdict(lambda: None)
    self.costs = defaultdict(lambda: np.inf)
    self.cost_dicts = defaultdict(lambda: {})

    init_sa = _params.init_stanceaction[mover.style]
    self.init_node = get_node_name('init', init_sa)
    self.final_node = get_node_name('final', init_sa)
    self.costs[self.init_node] = 0
    pass


  '''
    Graph functions
  '''
  def node_generator(self, stance: str, line_node: str):
    if line_node == 'final':
      yield self.final_node
    else:
      aug_line = self.line_nodes[line_node]['Line with active holds']
      for sa in self.stances.stanceaction_generator(stance, aug_line):
        yield get_node_name(line_node, sa)
    pass


  def edge_generator(self, node):
    line_node, sa, stance, d = self.full_parse(node)
    for line_node2 in self.line_edges[line_node]:
      for node in self.node_generator(stance, line_node2):
        yield node
    pass


  def edge_cost(self, node1, node2, verbose = False):
    '''
      TODO: Add options to control which dynamic costs we use
    '''
    line_node1, sa1, s1, d1 = self.full_parse(node1)
    line_node2, sa2, s2, d2 = self.full_parse(node2)
    if line_node2 == 'final':
      return {}
    time12 = self.timedelta(node1, node2)
    cost_dict = _memoizer.get_edge_cost(self.mover, 
        sa1, sa2, d1, d2, time12, line_node2, verbose=verbose)

    cost_dict['fast_jacks'] = 0

    # Dynamic cost functions
    node0 = self.predecessors[node1]
    if node0:
      line_node0, sa0, s0, d0 = self.full_parse(node0)
      time01 = self.timedelta(node0, node1)

      # Fast jacks dynamic cost function
      threshold = _params.jacks_footswitch_t_thresh
      if time01 < threshold and time12 < threshold:
        mv_cost01 = _memoizer.move_cost(self.mover, s0, s1, d0, d1)
        mv_cost12 = _memoizer.move_cost(self.mover, s1, s2, d1, d2)
        if mv_cost01 <= 0 and mv_cost12 <= 0:
          cost_dict['fast_jacks'] = self.mover.fast_jacks_cost(d0, d1, d2, time01, time12)
          self.stats_d['Dynamic cost: fast jacks'] += 1

      # Implement other dynamic costs here
      pass
    return cost_dict 


  def error_check(self, node1, node2):
    line_node1, sa1 = parse_node_name(node1)
    line_node2, sa2 = parse_node_name(node2)
    timedelta = self.timedelta(node1, node2)
    
    if timedelta < 0.001:
      output_log('Notes are too close together, likely from very high bpm')
      sys.exit(1)
    return


  '''
    Filtering
  '''
  def filter_node(self, line_node, sa):
    filters = [
      self.filter_node_beginner,
    ]
    return any([f(line_node, sa) for f in filters])


  def filter_node_beginner(self, line_node, sa):
    remove = False
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
    if not filters:
      return False
    return any([f(node1, node2) for f in filters])


  def filter_edge_unnecessary_jump(self, node1, node2):
    '''
      Get unnecessary jump flag by memoization
      Use strings as keys; dicts are not hashable
    '''
    line_node1, sa1, stance1, d1 = self.full_parse(node1)
    line_node2, sa2, stance2, d2 = self.full_parse(node2)
    line2 = self.line_node_to_line(line_node2)
    remove = _memoizer.unnecessary_jump(self.mover, stance1, stance2, d1, d2, line2)
    if remove:
      self.stats_d['Num. edges skipped by unnecessary jump'] += 1
    return remove


  '''
    Helper
  '''
  @functools.lru_cache(maxsize=None)
  def timedelta(self, node1, node2):
    ln1, sa1 = parse_node_name(node1)
    ln2, sa2 = parse_node_name(node2)
    return self.line_nodes[ln2]['Time'] - self.line_nodes[ln1]['Time']


  @functools.lru_cache(maxsize=None)
  def line_node_to_line(self, line_node):
    # '16.0' -> '00100'
    return self.line_nodes[line_node]['Line']


  @functools.lru_cache(maxsize=None)
  def full_parse(self, node: str):
    line_node, sa = parse_node_name(node)
    s = get_stance_from_sa(sa)
    d = self.parse_sa(sa)
    return line_node, sa, s, d


  '''
    Debugging
  '''
  def inspect_parents_of_node(self, node):
    # Why did dijkstra find parent X for node, and not parent Y?
    # Inspect costs of all parent nodes
    # TODO - This was broken by changes to node_generator. Fix up
    line_node, sa = parse_node_name(node)
    parent_lines = [l for l in self.line_nodes if line_node in self.line_edges[l]]
    parents = []
    for parent_line in parent_lines:
      for parent in self.node_generator(parent_line):
        cost = self.costs[parent]
        edge_cost = self.edge_cost(parent, node)
        parents.append((edge_cost, cost, parent))
    return sorted(parents)

  
  def inspect_nodes_by_line(self, line_name):
    return [node for node in self.costs if line_name in node]


  def interactive_debug(self):
    print(f'\nEntering interactive debug mode ...')
    print('''
      Intended workflow:
      - Manually open csv
      - Find a node to inspect
      Useful calls
        line_node, sa = parse_node_name(node)
        list(self.node_generator(prev_stance, line_node))
        self.inspect_parents_of_node(node)
        self.edge_cost(node1, node2, verbose=True)
        self.inspect_nodes_by_line(line_name)
      Ctrl+D to exit.
    ''')
    # node = '22.0:a9,a3;-1,--'
    # parent = '20.0:a7,a3;-1,--'
    # self.inspect_edge(parent, node)
    # parents = self.inspect_parents_of_node(node)
    import code; code.interact(local=dict(globals(), **locals()))
    return


'''
  Parsing
'''
@functools.lru_cache(maxsize=None)
def get_node_name(line_node: str, sa: str) -> str:
  return f'{line_node}:{sa}'


@functools.lru_cache(maxsize=None)
def parse_node_name(node: str):
  [line_node, sa] = node.split(':')
  return line_node, sa


@functools.lru_cache(maxsize=None)
def get_stance_from_sa(sa: str) -> str:
  return sa[:sa.index(';')]

