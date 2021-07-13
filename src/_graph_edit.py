import bisect
from collections import defaultdict
from genericpath import isdir

from pandas.core.indexes import multi

def edit(line_nodes, edges_out, edges_in):
  '''
    Edit graph of lines
    In spans of bracketable multis, remove regular nodes (force multi use)
    Pick a single graph path for multihits - pick earliest multi and skip overlapping multis (e.g., rolling center blue/yellow brackets will have an intermediate overlapping multi for center blues)
  '''
  norm_beats = [line_nodes[node]['Beat'] for node in line_nodes if 'multi' not in node]
  norm_beats = sorted(norm_beats)

  beat_to_multi = get_beat_to_multi(line_nodes)

  # Remove short chains of multis that are drills
  chains = get_multi_chains(line_nodes, edges_out)
  MIN_BRACKET_FOOTSWITCH_DRILL_LINES = 6
  num_drill_removed = 0
  for chain in chains:
    lines = [line_nodes[n]['Line with active holds'] for n in chain]
    if len(set(lines)) == 1 and len(lines) < MIN_BRACKET_FOOTSWITCH_DRILL_LINES:
      for n in chain:
        del line_nodes[n]
        num_drill_removed += 1
  print(f'Removed {num_drill_removed} multi lines for being short drills')

  # Remove regular nodes in spans of bracketable multis
  num_filtered = 0
  multi_to_covered_beats = {}
  multi_beats = [line_nodes[node]['Beat'] for node in line_nodes if 'multi' in node]
  for multi_beat in sorted(multi_beats):
    # Get immediate preceeding beat; assumes multis combine 2 lines
    prev_beat = get_prev_beat(norm_beats, multi_beat)
    replace_nodes = [str(prev_beat), str(multi_beat)]
    multi_to_covered_beats[multi_beat] = replace_nodes

    for node in replace_nodes:
      if node in line_nodes:
        del line_nodes[node]
        num_filtered += 1

    multi_nodes = beat_to_multi[multi_beat]
    for node in replace_nodes:
      if node in edges_in:
        for n in edges_in[node]:
          edges_out[n] += multi_nodes
          edges_out[n] = list(set(edges_out[n]))

  print(f'Filtered {num_filtered} lines covered by multis')
  return line_nodes, edges_out


'''
  Support
'''
def get_multi_chains(line_nodes, edges_out):
  multi_nodes = [n for n in line_nodes if 'multi' in n]
  chains = []
  import copy
  qu = copy.copy(multi_nodes)
  while qu:
    node = qu.pop(0)
    current_chain = [node]
    inner_qu = [n for n in edges_out[node] if 'multi' in n]
    assert len(inner_qu) < 2, 'Error: Assumption that each multi has at most one neighboring multi is broken'
    while inner_qu:
      curr = inner_qu.pop(0)
      current_chain.append(curr)
      inner_qu += [n for n in edges_out[curr] if 'multi' in n]
      assert len(inner_qu) < 2, 'Error: Assumption that each multi has at most one neighboring multi is broken'
    chains.append(current_chain)
    for node in current_chain:
      if node in qu:
        qu.remove(node)
  return chains


def get_prev_beat(sorted_beats, query):
  idx = bisect.bisect_left(sorted_beats, query)
  if idx - 1 >= 0:
    return sorted_beats[idx-1]
  else:
    return None


def get_beat_to_multi(line_nodes):
  beat_to_multi = defaultdict(list)
  for node in line_nodes:
    if 'multi' in node:
      beat = line_nodes[node]['Beat']
      beat_to_multi[beat].append(node)
  return beat_to_multi