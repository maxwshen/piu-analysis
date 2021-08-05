'''
  Construct timelines from line_df for front-end
'''
import numpy as np

def nps(df):
  # returns (x, y) coordinates
  # at time t, y is notes per second in t-1 to t
  maxt = int(round(max(df['Time'])))
  dp_df = df[df['Has downpress']]
  ts = list(dp_df['Time'])
  tdp = list(dp_df['Time since downpress'])
  
  xs, ys = [], []
  for t in range(1, maxt):
    num_popped = 0
    tdps = []
    while ts and ts[0] < t:
      ts.pop(0)
      tdps.append(tdp.pop(0))
      num_popped += 1
    xs.append(t)
    # Filter time sinces that are lower than 30 ms, which can happen with 1->2, e.g., Conflict S17
    filt_tdps = [t for t in tdps if t > 0.03]
    if filt_tdps:
      mean_timesince = 1/np.mean(filt_tdps)
      ys.append(mean_timesince)
    else:
      ys.append(0)
  return [xs, ys]


def binary_timeline(df, col, name):
  dfs = df[df[col]]
  xs = list(dfs['Time'])
  if xs:
    return [xs, [name]*len(xs)]
  else:
    return [[0], [name]]


def twist(df):
  hold_df = df[df['Twist angle'] != 'none']
  xs = list(hold_df['Time'])
  return [xs, ['Twist']*len(xs)]