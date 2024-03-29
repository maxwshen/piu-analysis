'''
  Graph for Dijkstra's.
  Nodes are stance-actions at specific lines.
'''
import sys, os, functools, itertools
import _config
import numpy as np, pandas as pd
from collections import defaultdict
from typing import List, Dict, Set, Tuple

import _memoizer, _params, _stances, segment

class Graph():
  def __init__(self, nm, mover, line_nodes, line_edges, annots, motifs):
    self.nm = nm
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
    self.cost_dicts = defaultdict(lambda: dict())

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

    self.hints = self.get_chart_hints(nm)
    self.has_hints = bool(self.hints is not None)
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
    Skip redundant cross-product between hands lines: too slow otherwise
  '''
  def detect_redundant_hands(self, prev_stance, prev_line, aug_line):
    prev_hands = len(prev_stance.split(',')) == 4
    onetotwo = prev_line.replace('1', '2') == aug_line
    twotothree = prev_line.replace('2', '3').replace('4', '3') == aug_line
    return prev_hands and (onetotwo or twotothree), onetotwo, twotothree


  def copy_hand_sa(self, prev_sa, onetotwo, twotothree):
    [prev_stance, prev_action] = prev_sa.split(';')
    if onetotwo:
      new_action = prev_action.replace('1', '2')
    if twotothree:
      new_action = prev_action.replace('2', '3').replace('4', '3')
    return ';'.join([prev_stance, new_action])


  '''
    Graph functions
  '''
  def node_generator(self, prev_node, line_node):
    '''
      Input: previous graph node (specific stance-action), current line node
      Propose stance actions, and filter using: 
      - Current tag path constraints
      - Filter redundant hands
    '''
    prev_line_node, prev_sa, prev_tag, prev_stance, prev_d = self.full_parse(prev_node)
    if prev_line_node != 'init':
      prev_line = self.line_nodes[prev_line_node]['Line with active holds']
    else:
      prev_line = ''
    aug_line = self.line_nodes[line_node]['Line with active holds']
    beat = self.line_nodes[line_node]['Beat']

    annot = self.annots[beat] if beat in self.annots else None
    motif = self.beat_to_motif[beat] if beat in self.beat_to_motif else None
    if motif is not None:
      motif_section, motif_annot = motif
      [motif_jfs, motif_twohits, motif_hold] = motif_annot.split('-')
      motif_start, motif_end = motif_section

    tag_jfs, tag_twohits, tag_hold = parse_tag(prev_tag)

    motif_branch = False
    jfs, twohits, hold = 'any', 'any', 'any'
    if motif:
      if beat == motif_start:
        motif_branch = True
      else:
        jfs = tag_jfs
        twohits = tag_twohits
        hold = tag_hold

    redundant_hands, onetotwo, twotothree = self.detect_redundant_hands(prev_stance, prev_line, aug_line)
    if redundant_hands:
      # Do not cross product hand stances -- too large
      copied_hand_sa = self.copy_hand_sa(prev_sa, onetotwo, twotothree)
      yield get_node_name(line_node, copied_hand_sa, f'{jfs}-{twohits}-{hold}')
    else:
      # Typical case
      sas = self.stances.stanceaction_generator(prev_stance, aug_line)

      if self.has_hints:
        sas = self.filter_sas_by_hints(sas, beat)
        ntags = [f'{jfs}-{twohits}-{hold}']*len(sas)
      else:
        if motif_branch:
          sas, ntags = self.motif_branch(prev_sa, sas, aug_line, annot,
              motif_jfs, motif_twohits, motif_hold)
        else:
          sas = self.filter_stanceactions(prev_sa, sas, aug_line, annot,
              jfs, twohits, hold)
          ntags = [f'{jfs}-{twohits}-{hold}']*len(sas)

      # Require use of brackets in multihits
      if 'multi' in line_node:
        sas, ntags = self.filter_to_brackets(sas, ntags)

      for sa, ntag in zip(sas, ntags):
        yield get_node_name(line_node, sa, ntag)


  def edge_generator(self, node):
    line_node, sa, tag, stance, d = self.full_parse(node)
    for line_node2 in self.line_edges[line_node]:
      if line_node2 in self.line_nodes:
        if line_node2 == 'final':
          yield self.final_node
        else:
          for node in self.node_generator(node, line_node2):
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
    if '1' not in line2:
      bad = False
      for c1, c2 in zip(line1, line2):
        if c2 == '2' and c1 not in list('13'):
          bad = True
        if c2 == '3' and c1 not in list('24'):
          bad = True
      if not bad:
        return

    if timedelta < 0.001:
      # ok for some charts like red swan s17
      pass
      # print(f'Error: Notes are too close together, likely from high bpm. {line1} {line2}')
      # import code; code.interact(local=dict(globals(), **locals()))
      # sys.exit(1)
    return


  '''
    Chart hints
    - Currently requires all lines to have an annotation, because switching back to default behavior in middle of tag is not implemented
  '''
  def get_chart_hints(self, nm):
    # hints[beat]['Left foot']
    hint_fn = _config.DATA_DIR + f'hints/{nm}.csv'
    if not os.path.isfile(hint_fn):
      return None

    print(f'Found chart hints - ignoring segment annotations and motifs')
    hint_df = pd.read_csv(hint_fn, index_col=0)

    def parse_hint(hint):
      # Read in as floats and np.nan. Convert to strings
      return '' if np.isnan(hint) else str(int(hint))

    left_hints = [parse_hint(hint) for hint in hint_df['Left foot hint']]
    right_hints = [parse_hint(hint) for hint in hint_df['Right foot hint']]
    beats = hint_df['Beat']

    hints = {}
    for beat, left_hint, right_hint in zip(beats, left_hints, right_hints):
      hints[round(beat, 3)] = (left_hint, right_hint)
    return hints


  def filter_sas_by_hints(self, sas, beat):
    def foot_hint_match(d, foot, hint):
      acts = d['limb_to_heel_action'][foot] + d['limb_to_toe_action'][foot]
      if hint == 'nan':
        return acts == '--'
      else:
        return all(x in acts for x in hint)

    feet = ['Left foot', 'Right foot']
    hints = self.hints[round(beat, 3)]
    new_sas = []
    for sa in sas:
      d = self.parse_sa(sa)
      ok = all(foot_hint_match(d, foot, hint) for foot, hint in zip(feet, hints))
      if ok:
        new_sas.append(sa)
    # print(f'{len(sas)}'.ljust(6), f'{len(new_sas)}')
    # import code; code.interact(local=dict(globals(), **locals()))
    return new_sas


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
              'jackoralternateorfree': ['jack', 'alternate', 'free'],
              'jackorfree': ['jack', 'free']}
    get_combo = lambda motif: combos.get(motif, [motif])

    out_sas, ntags = [], []
    for jfs in get_combo(motif_jfs):
      for twohits in get_combo(motif_twohits):
        for hold in get_combo(motif_hold):
          # Don't filter hold at branch
          if hold == 'jack':
            path_sas = self.filter_stanceactions(prev_sa, sas, aug_line, annot,
                jfs, twohits, hold)
          else:
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
    if '4' in aug_line or '3' in aug_line:
      if any(x in aug_line for x in list('12')):
        if hold in ['jack', 'alternate']:
          return self.filter_hold(prev_sa, sas, hold)
        elif hold == 'free':
          return sas
      else:
        return sas

    if annot in ['jack', 'footswitch', 'jackorfootswitch']:
      # annot takes priority
      if annot in ['jack', 'footswitch']:
        return self.filter_jackfootswitch(prev_sa, sas, annot)
      else:
        return self.filter_jackfootswitch(prev_sa, sas, jfs)
    elif annot in ['jump', 'bracket', 'jumporbracket']:
      if annot in ['jump', 'bracket']:
        return self.filter_twohits(sas, annot)
      else:
        return self.filter_twohits(sas, twohits)
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
    prev_limbs = [limb for limb in self.stances.limbs_doing(prev_sa, list('13'))]
    holding_limb = [limb for limb in self.stances.limbs_doing(prev_sa, list('24'))]
    if hold == 'jack':
      accept = lambda sa: all(x not in holding_limb 
          for x in self.stances.limbs_doing(sa, list('12')))
    elif hold == 'alternate':
      accept = lambda sa: all(x not in prev_limbs
          for x in self.stances.limbs_doing(sa, list('12')))
    else:
      accept = lambda sa: True
    return [sa for sa in sas if accept(sa)]


  def filter_to_brackets(self, sas, ntags):
    accept = lambda sa: len(self.stances.limbs_downpress(sa)) == 1
    ok = [i for i, sa in enumerate(sas) if accept(sa)]
    ok_sas = [sa for i, sa in enumerate(sas) if i in ok]
    ok_ntags = [ntag for i, ntag in enumerate(ntags) if i in ok]
    return ok_sas, ok_ntags


  '''
    Filter edges
  '''
  def filter_edge(self, node1, node2):
    # Ignore edges with heuristics
    filters = [
      self.filter_edge_unnecessary_jump,
      self.filter_jump_jump_crossover,
      self.filter_firstline_crossover,
      self.filter_firstline_unnecessary_bracketpos,
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


  def filter_jump_jump_crossover(self, node1, node2):
    '''
      If two lines are both jumps, second line should not have a crossover.
    '''
    if 'init' in node1:
      return False
    line_node1, sa1, tag1, stance1, d1 = self.full_parse(node1)
    line_node2, sa2, tag2, stance2, d2 = self.full_parse(node2)
    prev_jump = self.cost_dicts[node1]['jump'] > 0
    if prev_jump:
      if self.mover.foot_inversion_cost(d2) > 0:
        if self.mover.jump_cost(d1, d2) > 0:
          # print(f'Filtered jump jump crossover')
          return True
    return False


  def filter_firstline_crossover(self, node1, node2):
    # If first line is 'init', second line should not have a crossover.
    if 'init' not in node1:
      return False
    line_node1, sa1, tag1, stance1, d1 = self.full_parse(node1)
    line_node2, sa2, tag2, stance2, d2 = self.full_parse(node2)
    if self.mover.foot_inversion_cost(d2) > 0:
        return True
    return False
  

  def filter_firstline_unnecessary_bracketpos(self, node1, node2):
    # If first line is 'init', second line should use brackets unless required
    if 'init' not in node1:
      return False
    line_node1, sa1, tag1, stance1, d1 = self.full_parse(node1)
    line_node2, sa2, tag2, stance2, d2 = self.full_parse(node2)
    
    # If three or more heels/toes are pressing, then bracket is required
    actions = ''
    for foot in self.mover.feet:
      for key in ['limb_to_heel_action', 'limb_to_toe_action']:
        actions += d2[key][foot]
    if actions.count('1') + actions.count('2') > 2:
      return False

    # Otherwise, filter if using bracketpos
    for foot in self.mover.feet:
      if d2['limb_to_pos'][foot] in self.mover.bracket_pos:
        return True
    return False    


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

