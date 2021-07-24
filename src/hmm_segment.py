'''
'''

import numpy as np
from hmmlearn import hmm

# hmm parameters
MIN_COMP = 2
MAX_COMP = 4
NUM_REPS = 25

# smoothing - num. lines
max_short_len = 16

# muliplier on median time since to call breaks
median_fold_break = 8

'''
  HMM segmentation
'''
def transform_multinomial(data):
  # converts data into int indices
  unique_bs = set(data.ravel())
  b_to_i = {b: i for i, b in enumerate(unique_bs)}
  train_data = [b_to_i[b] for b in data]
  train_data = np.array(train_data).reshape(-1, 1)
  return train_data


def segment_multinomial_hmm(train_data):
  train_data = transform_multinomial(train_data)
  num_symbols = len(set(train_data.ravel()))

  best_zs = []
  best_aic = np.inf
  for nc in range(MIN_COMP, MAX_COMP):
    for rep in range(NUM_REPS):
      model = hmm.MultinomialHMM(n_components=nc)
      model.fit(train_data)

      zs = model.predict(train_data)
      log_prob = model.score(train_data)

      num_params = (nc-1) + nc*(nc-1) + (num_symbols-1)*nc
      aic = 2*num_params - 2*log_prob

      if aic < best_aic:
        best_aic = aic
        best_zs = zs

  return best_zs


'''
  Post-processing
'''
def smooth(segs):
  # replace first segment with second segment if too short
  bounds = seg_list_to_list_of_bounds(list(range(len(segs)+1)), segs)
  seg1_len = bounds[0][1] - bounds[0][0]
  if seg1_len <= max_short_len:
    seg2_id = segs[bounds[1][0]]
    segs = [seg2_id]*seg1_len + list(segs[seg1_len:])
    segs = np.array(segs)

  # replace last segment with second to last segment if too short
  last_len = bounds[-1][1] - bounds[-1][0]
  if last_len <= max_short_len:
    segl2_id = segs[bounds[-2][0]]
    segs = list(segs[:-last_len]) + [segl2_id]*last_len
    segs = np.array(segs)

  # simple for now, detect isolated short runs and replace them with earlier seg
  num_replaced = 0
  for short_len in range(max_short_len, -1, -1):
    window = short_len + 2
    for i in range(len(segs) - window):
      [first, *mid, last] = segs[i : i + window]
      if len(set(mid)) == 1:
        mid_item = mid[0]
        if mid_item != first and mid_item != last:
          # replace
          for k in range(i+1, i+window-1):
            segs[k] = first
            num_replaced += 1
  print(f'Replaced {num_replaced} items by smoothing')
  return segs


def combine_bounds(bounds, max_indiv_len = 16):
  # max_indiv_len: if greater than this, do not combine
  new_bounds = []
  comb_to_indiv = {}
  i = 0
  bound_len = lambda b: b[1] - b[0]

  while i < len(bounds):
    if bound_len(bounds[i]) <= max_indiv_len:
      j = i + 1
      while j < len(bounds) and bound_len(bounds[j]) <= max_indiv_len:
        j += 1
      cbound = (bounds[i][0], bounds[j-1][1])
      comb_to_indiv[len(new_bounds)] = list(range(i, j))
      new_bounds.append(cbound)
      i = j
    else:
      comb_to_indiv[len(new_bounds)] = [i]
      new_bounds.append(bounds[i])
      i += 1

  return new_bounds, comb_to_indiv


def split_breaks(segs, time_sinces, dp_idxs):
  # Split segments by breaks (large time gaps)
  bounds = seg_list_to_list_of_bounds(dp_idxs, segs)

  bs = []
  for bound in bounds:
    ts = time_sinces[bound[0] : bound[1]]
    break_threshold = np.median(ts) * median_fold_break
    break_idxs = [i for i, t in enumerate(ts) if t > break_threshold]
    new_bounds = [bound[0]] + [bound[0] + bi for bi in break_idxs] + [bound[1]]

    for i in range(1, len(new_bounds)):
      bs.append((new_bounds[i-1], new_bounds[i]))
  
  return list_of_bounds_to_seg_list(bs)


'''
  Data structure
'''
def seg_list_to_list_of_bounds(idxs, segs):
  # List of N ints -> List of segments by boundary changes
  i = 0
  bounds = []
  while i < len(segs):
    j = i + 1
    while j < len(segs) and segs[j] == segs[i]:
      j += 1
    bounds.append((idxs[i], idxs[j]))
    i = j
  return bounds


def list_of_bounds_to_seg_list(bs):
  # List of segments by boundary changes -> list of N ints
  res = []
  for i, b in enumerate(bs):
    res += [i] * (b[1] - b[0])
  return res


'''
  Primary
'''
def segment(df, num_segments=10):
  '''
    Segments df by time since downpress
    Returns a list of sub dataframes
  '''
  dp_df = df[df['Has downpress']]
  data = dp_df['Time since downpress']
  data = np.array([round(bs, 2) for bs in data])
  dp_idxs = list(dp_df.index) + [len(df)]
  
  segs = segment_multinomial_hmm(data)
  segs = split_breaks(segs, data, list(range(len(data)+1)))
  smooth_segs = smooth(segs)

  # bounds are used for summary plotting
  bounds = seg_list_to_list_of_bounds(dp_idxs, smooth_segs)
  # print(bounds)
  print(f'Found {len(bounds)} groups')

  # groups (less than 10) are used for chart detail plotting
  groups = bounds
  num_segs = len(groups)
  if num_segs > num_segments:
    max_indiv_len = 4
    while num_segs > num_segments:
      cbs, comb_to_indiv = combine_bounds(bounds, max_indiv_len)
      num_segs = len(cbs)
      max_indiv_len += 1
    groups = cbs
    print(f'Combined into {len(groups)} groups with max. indiv. len {max_indiv_len}')
  else:
    groups = bounds
    comb_to_indiv = {i: [i] for i in range(len(groups))}

  all_dfs = []
  for bound in bounds:
    dfs = df.iloc[bound[0]:bound[1]]
    all_dfs.append(dfs)
  
  return all_dfs, groups, comb_to_indiv
