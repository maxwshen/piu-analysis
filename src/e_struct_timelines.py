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
      tdps = tdp.pop(0)
      num_popped += 1
    xs.append(t)
    if tdps:
      ys.append(1/np.mean(tdps))
    else:
      ys.append(0)
  return [xs, ys]


def binary_timeline(df, col, name):
  dfs = df[df[col]]
  xs = list(dfs['Time'])
  return [xs, [name]*len(xs)]


def twist(df):
  hold_df = df[df['Twist angle'] != 'none']
  xs = list(hold_df['Time'])
  return [xs, ['Twist']*len(xs)]