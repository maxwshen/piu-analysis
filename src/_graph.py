'''
  Graph for Dijkstra's.
  Nodes are stance-actions at specific lines.
'''
import sys, functools, itertools
import numpy as np, pandas as pd
from collections import defaultdict
from typing import List, Dict, Set, Tuple

import _memoizer, _params, _stances, segment

class Graph():
  def __init__(self, mover, line_nodes, line_edges, annots, motifs):
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
    default_tag = 'any-any-any'
    self.init_node = get_node_name('init', init_sa, default_tag)
    self.final_node = get_node_name('final', init_sa, default_tag)
    self.costs[self.init_node] = 0

    self.path_consistent_annotations = set(['jackorfootswitch', 'jumporbracket'])
    self.path_consistent_tags = set(['jack', 'footswitch', 'jump', 'bracket'])

    self.beats = [self.line_nodes[x]['Beat'] for x in self.line_nodes]
    self.annots, self.motifs = segment.filter_annots(self.beats, annots, motifs)
    self.beat_to_motif = self.get_beat_to_motif()
    self.beat_to_motiflen = self.get_beat_to_motif_len()
    pass


  def get_beat_to_motif(self):
    # beat -> section, motif_annotation
    beat_to_motif = {}
    for beat in self.beats:
      for section in self.motifs:
        if segment.beat_in_section(beat, section):
          motif_annot = self.motifs[section]
          beat_to_motif[beat] = (section, motif_annot)
          break
    return beat_to_motif


  def get_beat_to_motif_len(self):
    beat_to_motiflen = {}
    for beat in self.beats:
      for section in self.motifs:
        if segment.beat_in_section(beat, section):
          motif_len = len([b for b in self.beats
                          if section[0] <= b <= section[1]])
          beat_to_motiflen[beat] = motif_len
          break
    return beat_to_motiflen


  '''
    Graph functions
  '''
  def node_generator(self, stance, tag, prev_sa, line_node):
    aug_line = self.line_nodes[line_node]['Line with active holds']
    beat = self.line_nodes[line_node]['Beat']

    annot = self.annots[beat] if beat in self.annots else None
    motif = self.beat_to_motif[beat] if beat in self.beat_to_motif else None
    if motif is not None:
      motif_section, motif_annot = motif
      [motif_jfs, motif_twohits, motif_hold] = motif_annot.split('-')
      motif_start, motif_end = motif_section

    tag_jfs, tag_twohits, tag_hold = parse_tag(tag)

    motif_branch = False
    jfs, twohits, hold = 'any', 'any', 'any'
    if motif:
      if beat == motif_start:
        motif_branch = True
      else:
        jfs = tag_jfs
        twohits = tag_twohits
        hold = tag_hold
    
    sas = self.stances.stanceaction_generator(stance, aug_line)
    if motif_branch:
      sas, ntags = self.motif_branch(prev_sa, sas, aug_line, annot,
          motif_jfs, motif_twohits, motif_hold)
    else:
      sas = self.filter_stanceactions(prev_sa, sas, aug_line, annot,
          jfs, twohits, hold)
      ntags = [f'{jfs}-{twohits}-{hold}']*len(sas)

    for sa, ntag in zip(sas, ntags):
      yield get_node_name(line_node, sa, ntag)


  def edge_generator(self, node):
    line_node, sa, tag, stance, d = self.full_parse(node)
    for line_node2 in self.line_edges[line_node]:
      if line_node2 == 'final':
        yield self.final_node
      else:
        for node in self.node_generator(stance, tag, sa, line_node2):
          yield node


  def edge_cost(self, node1, node2, verbose = False):
    line_node1, sa1, tag1, s1, d1 = self.full_parse(node1)
    line_node2, sa2, tag2, s2, d2 = self.full_parse(node2)
    if line_node2 == 'final':
      return {}
    time12 = self.timedelta(node1, node2)
    beat = self.line_nodes[line_node2]['Beat']
    motif_len = self.beat_to_motiflen.get(beat, None)
    cost_dict = _memoizer.get_edge_cost(self.mover, 
        sa1, sa2, d1, d2, time12, line_node2,
        self.line_nodes[line_node2]['Line'], tag1, tag2,
        motif_len, verbose=verbose)
    return cost_dict 


  def error_check(self, node1, node2):
    line_node1, sa1, tag1 = parse_node_name(node1)
    line_node2, sa2, tag2 = parse_node_name(node2)
    ok = ['init', 'final']
    if line_node1 in ok or line_node2 in ok:
      return
    line1 = self.line_nodes[line_node1]['Line with active holds']
    line2 = self.line_nodes[line_node2]['Line with active holds']
    timedelta = self.timedelta(node1, node2)
    
    # Forgive fast 1/3->2 and fast 2/4->3
    bad = False
    for c1, c2 in zip(line1, line2):
      if c2 == '2' and c1 not in list('13'):
        bad = True
      if c2 == '3' and c1 not in list('24'):
        bad = True

    if not bad:
      return

    if timedelta < 0.001:
      print(f'ERROR: Notes are too close together, likely from high bpm. {line1} {line2}')
      import code; code.interact(local=dict(globals(), **locals()))
      sys.exit(1)
    return


  '''
    Filter stance actions - constrain Dijkstra paths
  '''
  def motif_branch(self, prev_sa, sas, aug_line, annot,
      motif_jfs, motif_twohits, motif_hold):
    '''
      At the beginning of a motif, create branching parallel paths
      with tag.
      For each branch, propose stance-actions compatible with first line
    '''
    combos = {'jackorfootswitch': ['jack', 'footswitch'],
              'jumporbracket': ['jump', 'bracket'],
              'jackoralternateorfree': ['jack', 'alternate', 'free']}
    get_combo = lambda motif: combos.get(motif, [motif])

    out_sas, ntags = [], []
    for jfs in get_combo(motif_jfs):
      for twohits in get_combo(motif_twohits):
        for hold in get_combo(motif_hold):
          # Don't filter hold at branch
          path_sas = self.filter_stanceactions(prev_sa, sas, aug_line, annot,
              jfs, twohits, 'any')
          out_sas += path_sas
          ntags += [f'{jfs}-{twohits}-{hold}']*len(path_sas)
    return out_sas, ntags


  def filter_stanceactions(self, prev_sa, sas, aug_line, annot, jfs, twohits, hold):
    '''
      Filters stanceactions based on flags.
        annot: ['', 'jackorfootswitch', 'alternate', 'jumporbracket', 'jump']
        jfs (jack/footswitch): ['any', 'jack', 'footswitch']
        twohits: ['any', 'jump',' 'bracket']
        hold: ['any', 'jack', 'alternate', 'free']
      Returns filtered sas and list of tags
    '''
    if '4' in aug_line:
      if any(x in aug_line for x in list('12')):
        if hold in ['jack', 'alternate']:
          return self.filter_hold(prev_sa, sas, hold)
        elif hold == 'free':
          return sas
      else:
        return sas

    if annot in ['jack', 'footswitch', 'jackorfootswitch']:
      return self.filter_jackfootswitch(prev_sa, sas, jfs)
    elif annot in ['jump', 'bracket', 'jumporbracket']:
      return self.filter_twohits(sas, annot)
    elif annot == 'alternate':
      return self.filter_alternate(prev_sa, sas)
    elif annot == 'same':
      return self.filter_same(prev_sa, sas)
    return sas


  def filter_jackfootswitch(self, prev_sa, sas, jfs):
    prev_limbs = self.stances.limbs_downpress(prev_sa)
    if jfs == 'jack':
      accept = lambda sa: self.stances.limbs_downpress(sa) == prev_limbs
    elif jfs == 'footswitch':
      accept = lambda sa: self.stances.limbs_downpress(sa) != prev_limbs
    elif jfs in ['jackorfootswitch', 'any']:
      accept = lambda sa: True
    else:
      print(f'ERROR: jfs is {jfs}')
    return [sa for sa in sas if accept(sa)]


  def filter_twohits(self, sas, twohits):
    if twohits == 'jump':
      def accept(sa):
        two_feet = len(self.stances.limbs_downpress(sa)) == 2
        d = self.parse_sa(sa)
        not_bracket = True
        for foot in ['Left foot', 'Right foot']:
          if d['limb_to_pos'][foot] in self.mover.bracket_pos:
            not_bracket = False
        return two_feet and not_bracket
    elif twohits == 'bracket':
      accept = lambda sa: len(self.stances.limbs_downpress(sa)) == 1
    elif twohits in ['jumporbracket', 'any']:
      accept = lambda sa: True
    else:
      print(f'ERROR: twohits is {twohits}')
    return [sa for sa in sas if accept(sa)]


  def filter_alternate(self, prev_sa, sas):
    prev_limbs = self.stances.limbs_downpress(prev_sa)
    accept = lambda sa: self.stances.limbs_downpress(sa) != prev_limbs
    return [sa for sa in sas if accept(sa)]


  def filter_same(self, prev_sa, sas):
    prev_limbs = self.stances.limbs_downpress(prev_sa)
    accept = lambda sa: self.stances.limbs_downpress(sa) == prev_limbs
    return [sa for sa in sas if accept(sa)]


  def filter_hold(self, prev_sa, sas, hold):
    '''
      TODO - Bigger problem is that I assumed implicitly that hold motifs would only start at the first downpress and end.
      Right now, hold motifs are longer than assumed, which is why these bugs are happening.
    '''
    prev_limbs = [limb for limb in self.stances.limbs_doing(prev_sa, list('13'))]
    holding_limb = [limb for limb in self.stances.limbs_doing(prev_sa, list('4'))]
    if hold == 'jack':
      accept = lambda sa: self.stances.limbs_doing(sa, list('12')) != holding_limb
    elif hold == 'alternate':
      accept = lambda sa: all(x not in prev_limbs
          for x in self.stances.limbs_doing(sa, list('12')))
    else:
      accept = lambda sa: True
    return [sa for sa in sas if accept(sa)]


  '''
    Filter edges
  '''
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
    line_node1, sa1, tag1, stance1, d1 = self.full_parse(node1)
    line_node2, sa2, tag2, stance2, d2 = self.full_parse(node2)
    line2 = self.line_node_to_line(line_node2)
    remove = _memoizer.unnecessary_jump(self.mover,
        stance1, stance2, d1, d2, line2)
    if remove:
      self.stats_d['Num. edges skipped by unnecessary jump'] += 1
    return remove

  
  '''
    Helper
  '''
  @functools.lru_cache(maxsize=None)
  def timedelta(self, node1, node2):
    ln1, sa1, tag1 = parse_node_name(node1)
    ln2, sa2, tag2 = parse_node_name(node2)
    return self.line_nodes[ln2]['Time'] - self.line_nodes[ln1]['Time']


  @functools.lru_cache(maxsize=None)
  def line_node_to_line(self, line_node):
    # '16.0' -> '00100'
    return self.line_nodes[line_node]['Line']


  @functools.lru_cache(maxsize=None)
  def full_parse(self, node: str):
    line_node, sa, tag = parse_node_name(node)
    s = get_stance_from_sa(sa)
    d = self.parse_sa(sa)
    return line_node, sa, tag, s, d


  '''
    Debugging
  '''
  def inspect_parents_of_node(self, node):
    # Why did dijkstra find parent X for node, and not parent Y?
    # Inspect costs of all parent nodes
    # TODO - This was broken by changes to node_generator. Fix up
    line_node, sa, tag = parse_node_name(node)
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
        line_node, sa, tag = parse_node_name(node)
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
def get_node_name(line_node, sa, tag) -> str:
  return ':'.join([line_node, sa, tag])


@functools.lru_cache(maxsize=None)
def parse_node_name(node: str):
  [line_node, sa, tag] = node.split(':')
  return line_node, sa, tag


@functools.lru_cache(maxsize=None)
def parse_tag(tag):
  [tag_jfs, tag_twohit, tag_hold] = tag.split('-')
  return tag_jfs, tag_twohit, tag_hold


@functools.lru_cache(maxsize=None)
def get_stance_from_sa(sa: str) -> str:
  return sa[:sa.index(';')]

