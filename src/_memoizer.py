'''
  Custom memoization cache functions.
  
  _movement functions mostly take in parsed_stanceaction dicts as input,
  which are parsed from strings in _movement and c_dijkstra.

  Here, eval functions using dicts, but cache using strings.
  Sometimes, we can cache using subset of string, increasing cache hit rate.
'''
from collections import defaultdict


jump_memoizer = defaultdict(lambda: 0)
def unnecessary_jump(mover, s1, s2, d1, d2, child_line):
  key = (s1, s2, child_line)
  if key in jump_memoizer:
    jump_flag = jump_memoizer[key]
    jump_memoizer['Num hits'] += 1
  else:
    jump_flag = mover.unnecessary_jump(d1, d2, child_line)
    jump_memoizer[key] = jump_flag
  return jump_flag


cost_memoizer = defaultdict(lambda: 0)
def get_edge_cost(mover, sa1, sa2, d1, d2, timedelta, child):
  key = (sa1, sa2, timedelta)
  if key in cost_memoizer:
    edge_cost = cost_memoizer[key]
    cost_memoizer['Num hits'] += 1
  else:
    edge_cost = mover.get_cost_from_ds(d1, d2, time=timedelta)

    # Modify cost for memoization
    # Multihit modifier if brackets
    multi_mod = mover.multihit_modifier(d1, d2, child)
    edge_cost += multi_mod

    # Apply time cost here to get memoization speedup and time sensitivity
    if 0.001 < timedelta < mover.costs['Time threshold']:
      time_factor = mover.costs['Time normalizer'] / timedelta
      edge_cost *= time_factor
  return edge_cost


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
def beginner_flag(mover, s, d):
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
  created at the top of c_dijkstra.
  This setup is slightly preferable -- keeps c_dijkstra shorter.
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
    'Jump': jump_memoizer,
    'Cost': cost_memoizer,
    'Beginner': beginner_memoizer,
  }
  for nm, memo in memoizers.items():
    stats_d[f'{nm} cache, num hits'] = memo['Num hits']
    stats_d[f'{nm} cache, size'] = len(memo)
  return stats_d