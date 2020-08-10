# 
from __future__ import division
import _config, _data, _stances, util, pickle, _params
import sys, os, fnmatch, datetime, subprocess
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from more_itertools import unique_everseen

import _movement

# Default params
inp_dir = _config.OUT_PLACE + f'b_graph/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

##
# Functions
##
def dijkstra(nodes, edges_out, edges_in):
  '''
    nodes[node_nm] = {
      'Time': float,
      'Beat': float,
      'Line': str,
      'Line with active holds': str,
      'Measure': int,
      'BPM': float,
      'Stance actions': List[str],
      'Previous panels': List[str],
      'Best parent': node_nm: str; filled in during Dijkstra's, backtrack to find best path
    }
    edges = {
      node_nm: List[node_nm: str]
    }
  
    Initial node has extra keys
    nodes['init']['Steptype'] = singles or doubles
    nodes['init']['Timing judge']
  '''
  steptype = nodes['init']['Steptype']
  mover = _movement.Movement(style = steptype)

  node_qu = ['init']

  while len(node_qu) > 0:
    nm, node_qu = node_qu[0], node_qu[1:]
    children = edges_out[nm]
    node_qu += children

    sa1 = nodes[nm]['Stance actions'][0]
    sa2 = nodes[children[0]]['Stance actions'][0]
    timedelta = nodes[children[0]]['Time'] - nodes[nm]['Time']
    cost = mover.get_cost(sa1, sa2, time = timedelta)
    print(sa1, sa2, cost)

    import code; code.interact(local=dict(globals(), **locals()))

  return



'''
  Helper
'''
def load_data(nm):
  with open(inp_dir + f'{nm}.pkl', 'rb') as f:
    nodes, edges_out, edges_in = pickle.load(f)
  return nodes, edges_out, edges_in


##
# qsub
##
def gen_qsubs():
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  num_scripts = 0
  for idx in range(0, 10):
    command = 'python %s.py %s' % (NAME, idx)
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + 'q_%s_%s.sh' % (script_id, idx)
    with open(sh_fn, 'w') as f:
      f.write('#!/bin/bash\n%s\n' % (command))
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append('qsub -j y -V -wd %s %s' % (_config.SRC_DIR, sh_fn))

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print('Wrote %s shell scripts to %s' % (num_scripts, qsubs_dir))
  return


##
# Main
##
@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  nm = 'Super Fantasy - SHK S19 arcade'
  # nm = 'Sorceress Elise - YAHPP S23 arcade'

  nodes, edges_out, edges_in = load_data(nm)
  dijkstra(nodes, edges_out, edges_in)
  return


if __name__ == '__main__':
  main()