import bisect
from collections import defaultdict

from pandas.core.indexes import multi

def edit(line_nodes, edges_out, edges_in):
  '''
    Edit graph of lines
    In spans of bracketable multis, remove regular nodes (force multi use)
    Pick a single graph path for multihits - pick earliest multi and skip overlapping multis (e.g., rolling center blue/yellow brackets will have an intermediate overlapping multi for center blues)
  '''
  norm_beats = [line_nodes[node]['Beat'] for node in line_nodes if 'multi' not in node]
  multi_beats = [line_nodes[node]['Beat'] for node in line_nodes if 'multi' in node]
  norm_beats = sorted(norm_beats)

  beat_to_multi = get_beat_to_multi(line_nodes)

  num_filtered = 0
  multi_to_covered_beats = {}
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