'''
  Featurize charts for clustering
'''
import _config, util
import sys, os, pickle, fnmatch, datetime, subprocess, functools, re
import numpy as np, pandas as pd
from collections import defaultdict, Counter

import _qsub
import cluster_featurize, chart_compare

# Default params
inp_dir = _config.OUT_PLACE + 'cluster_featurize/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

from scipy.spatial import KDTree

POWER = 1
HIGH_PCT_THRESHOLD = 0.65**POWER
LOWER_LVL = 3
UPPER_LVL = 2
NUM_RECS_PER_LVL = 3

cluster_feature_df = pd.read_csv(inp_dir + f'cluster_features.csv', index_col=0)

def get_feature_mat():
  df = cluster_feature_df
  ft_cols = [c for c in df.columns if 'Feature - ' in c]
  ignore = [
    'Feature - Hold',
    'Feature - Hold taps',
    'Feature - Hold run',
    'Feature - Irregular rhythm',
    'Feature - Rhythm change',
  ]
  ft_cols = [col for col in ft_cols if col not in ignore]
  ft_mat = np.array(df[ft_cols])
  ft_mat = ft_mat**POWER
  return ft_mat, ft_cols


def drop_redundant_features(ft_cols, jdxs):
  fts = [ft_cols[jdx] for jdx in jdxs]
  redundant = [
    ['Mid4 doubles', 'Mid6 doubles']
  ]
  new_jdxs = jdxs
  for red in redundant:
    if all(f'Feature - {x}' in fts for x in red):
      drop_idx = ft_cols.index(f'Feature - {red[-1]}')
      new_jdxs.remove(drop_idx)
  return new_jdxs


def sanitize(query_ft_col):
  # Reformat feature columns for front-end
  replace = {
    'Twist angle - none': 'No twists',
  }
  tag = query_ft_col.replace('Feature - ', '')
  tag = chart_compare.rename_tag(tag)
  for rk, rv in replace.items():
    tag = tag.replace(rk, rv)
  return tag


def get_neighbors(nm, verbose = False):
  df = cluster_feature_df
  ft_mat, ft_cols = get_feature_mat()
  nms = list(df['Name (unique)'])
  levels = list(df['Level'])
  idx = nms.index(nm)

  # Compare only on high-percentile features
  # In featurize, percentiles are 0'd out of feature frequency is <5%
  high_pct_jdxs = [i for i,v in enumerate(ft_mat[idx]) if v >= HIGH_PCT_THRESHOLD]
  high_pct_jdxs = drop_redundant_features(ft_cols, high_pct_jdxs)
  high_pct_ft_mat = ft_mat[:, high_pct_jdxs]

  query_ft_cols = [sanitize(ft_cols[j]) for j in high_pct_jdxs]
  
  kdt = KDTree(high_pct_ft_mat)
  
  # Query using [1] vector, not true query features
  q = [1] * len(high_pct_jdxs)
  dists, ne_idxs = kdt.query(q, k=len(ft_mat)//10)

  query_lvl = levels[idx]
  in_range = lambda neighbor_level: query_lvl - LOWER_LVL <= neighbor_level <= query_lvl + UPPER_LVL
  num_added = 0
  total_recs = NUM_RECS_PER_LVL * (UPPER_LVL + LOWER_LVL + 1)
  top3s = defaultdict(list)
  for dist, ne_idx in zip(dists[1:], ne_idxs[1:]):
    target_level = levels[ne_idx]
    if in_range(target_level):
      neighbor_nm = nms[ne_idx]
      if len(top3s[target_level]) < NUM_RECS_PER_LVL:
        if 'hidden' not in neighbor_nm and neighbor_nm != nm:
          # Avoid 'v1/v2/v3' charts - usually not 
          if 'v' not in neighbor_nm[-3:]:
            top3s[target_level].append(neighbor_nm)
            num_added += 1
    if num_added > total_recs:
      break

  sorted_lvls = sorted(list(top3s.keys()))
  top3s = {k: top3s[k] for k in sorted_lvls}
  if verbose:
    print(nm)
    print(query_ft_cols)
    for k, v in top3s.items():
      print(k, v)
  return query_ft_cols, top3s


def test():
  nm = 'Super Fantasy - SHK S16 arcade'
  # nm = 'Trashy Innocence - Last Note. D15 arcade'
  get_neighbors(nm, verbose=True)
  return

if __name__ == '__main__':
  test()