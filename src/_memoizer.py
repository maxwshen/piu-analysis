'''
  Custom memoization cache functions.
  
  _movement functions mostly take in parsed_stanceaction dicts as input,
  which are parsed from strings in _movement and c_dijkstra.

  Here, eval functions using dicts, but cache using strings.
  Sometimes, we can cache using subset of string, increasing cache hit rate.
'''
import copy
from collections import defaultdict
import _notelines, _graph

cost_memoizer = defaultdict(lambda: 0)
def get_edge_cost(mover, sa1, sa2, d1, d2, timedelta,
    line_node2, line, tag1, tag2, motif_len, verbose=False):
  key = (sa1, sa2)
  if key in cost_memoizer and not verbose:
    cost_dict = cost_memoizer[key]
    cost_memoizer['Num hits'] += 1
  else:
    cost_dict = mover.get_cost_from_ds(d1, d2, verbose=verbose)
    cost_memoizer[key] = cost_dict

  # Modify based on data outside high-percentage memoizer keys
  cost_dict = copy.copy(cost_dict)
  cost_dict['multihit'] = mover.multihit_modifier(d1, d2, line_node2)

  # if 0.001 < timedelta < mover.params['Time threshold']:
  #   time_factor = max(1.0, mover.params['Time normalizer'] / timedelta)
  #   time_affected = (
  #     # 'double_step',
  #     # 'bracket',
  #     # 'move_without_action',
  #     # 'jump',
  #   )
  #   for prop in time_affected:
  #     cost_dict[prop] *= time_factor
  # elif timedelta > mover.params['Time threshold']:
  #   # cost_dict['double_step'] = 0
  #   pass

  # Penalize using bracket positions for single-panel lines
  cost_dict['bracket_for_1panel'] = mover.bracket_on_singlepanel_line(d2, 
      line)

  # Penalize alternating feet on holds
  cost_dict['hold_alternate'] = mover.hold_alternate(tag1, tag2, motif_len)

  if verbose:
    print(f'Edge cost: {cost_dict}')
  return cost_dict


jump_memoizer = defaultdict(lambda: 0)
def unnecessary_jump(mover, s1, s2, d1, d2, line_node2):
  key = (s1, s2, line_node2)
  if key in jump_memoizer:
    jump_flag = jump_memoizer[key]
    jump_memoizer['Num hits'] += 1
  else:
    jump_flag = mover.unnecessary_jump(d1, d2, line_node2)
    jump_memoizer[key] = jump_flag
  return jump_flag


move_memoizer = defaultdict(lambda: 0)
def move_cost(mover, s1, s2, d1, d2):
  key = (s1, s2)
  if key in move_memoizer:
    mv_cost = move_memoizer[key]
    move_memoizer['Num hits'] += 1
  else:
    mv_cost = mover.move_cost(d1, d2)
    move_memoizer[key] = mv_cost
  return mv_cost


beginner_memoizer = defaultdict(lambda: 0)
def beginner_ok(mover, s, d):
  key = s
  if key in beginner_memoizer:
    beginner_flag = beginner_memoizer[key]
    beginner_memoizer['Num hits'] += 1
  else:
    beginner_flag = mover.beginner_ok(d)
    beginner_memoizer[key] = beginner_flag
  return beginner_flag


'''
  TODO - Consider using template to make code more concise.
  m needs to be instantiated per function to memoize though,
  which means this function should be in a class, with instances
  created at the top of any other script that needs this.
'''
m = defaultdict(lambda: 0)
def generic(mover, key, args, func):
  if key in m:
    res = m[key]
    m['Num hits'] += 1
  else:
    res = func(d)
    m[key] = res
  return res


'''
  Memoization stats
'''
def add_cache_stats(nm, cached_func, stats_d):
  info = cached_func.cache_info()
  stats_d[f'{nm} cache, num hits'] = info[0]
  stats_d[f'{nm} cache, size'] = info[-1]
  return stats_d


def add_custom_memoizer_stats(stats_d):
  memoizers = {
    'Move': move_memoizer,
    'Cost': cost_memoizer,
    'Beginner': beginner_memoizer,
  }
  for nm, memo in memoizers.items():
    stats_d[f'{nm} cache, num hits'] = memo['Num hits']
    stats_d[f'{nm} cache, size'] = len(memo)
  return stats_d