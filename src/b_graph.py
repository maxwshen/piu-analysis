# 
from __future__ import division
import _config, _data, _stances, util, pickle, _params
import sys, os, fnmatch, datetime, subprocess
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from typing import List, Dict, Set, Tuple
from more_itertools import unique_everseen

# Default params
inp_dir_a = _config.OUT_PLACE + f'a_format_data/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

# Load data
all_notes = pickle.load(open(inp_dir_a + f'notes.pkl', 'rb'))
all_bpms = pickle.load(open(inp_dir_a + f'bpms.pkl', 'rb'))

sc_df = pd.read_csv(inp_dir_a + f'all_stepcharts.csv', index_col = 0)

# Stances
stance_store = {
  'singles': _stances.Stances(style = 'singles'),
  # 'doubles': _stances.Stances(style = 'doubles'),
}

##
# Functions
##
def form_graph(nm: str):
  '''
    Assumes 4/4 time signature (other time sigs are extremely rare, and assumed to be filtered out)

    Notes format: https://github.com/stepmania/stepmania/wiki/sm.
  '''
  try:
    atts = sc_df[sc_df['Name (unique)'] == nm].iloc[0]
  except IndexError:
    print(f'ERROR: Failed to find stepchart {nm}')
    sys.exit(1)

  if 'S' in atts['Steptype simple']:
    steptype = 'singles'
  elif 'D' in atts['Steptype simple']:
    steptype = 'doubles'
  else:
    assert False, 'No stance found' 
  stance = stance_store[steptype]

  notes = all_notes[nm]
  measures = [s.strip() for s in notes.split(',')]
  bpms = parse_bpm(all_bpms[nm])
  beats_per_measure = 4

  # Testing -- first 9 measures only
  # measures = measures[:9]
  measures = measures[:26]

  beat = 0
  time = 0     # units = seconds
  bpm = None
  prev_presses = []
  active_holds = set()

  # Init bpm at beat = 0
  if beat >= bpms[0][0]:
    # print(beat, bpms)
    bpm = bpms[0][1]
    bpms = bpms[1:]
  assert bpm is not None, 'Failed to set bpm'

  nodes = dict()
  edges_out = defaultdict(list)
  edges_in = defaultdict(list)

  nodes['init'] = {
    'Time': time,
    'Beat': beat,
    'Measure': 0,
    'BPM': bpm,
    'Stance actions': stance.initial_stanceaction(),
    'Previous panels': [],
    'Steptype': steptype,
    'Timing judge': '',
  }
  prev_node_nm = 'init'
  edges_in['init'] = []

  timer = util.Timer(total = len(measures))
  for measure_num, measure in enumerate(measures):
    lines = measure.split('\n')
    num_subbeats = len(lines)
    note_type = num_subbeats
    for line in lines:
      beat_increment = beats_per_measure / num_subbeats

      if has_notes(line):
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
          }
          edges = {
            node_nm: List[node_nm: str]
          }
        '''

        # Add active holds into line as 4
        # 01000 -> 01040
        aug_line = list(line)
        for panel in active_holds:
          idx = stance.panel_to_idx[panel]
          if aug_line[idx] == '0':
            aug_line[idx] = '4'
        aug_line = ''.join(aug_line)

        active_panel_to_action = stance.text_to_panel_to_action(line)
        # prev_panels = list(active_holds) + prev_presses
        prev_panels = prev_presses
        sas = stance.get_stanceactions(aug_line, prev_panels = prev_panels)

        node_nm = f'{beat}'
        nodes[node_nm] = {
          'Time': time,
          'Beat': beat,
          'Line': line,
          'Line with active holds': aug_line,
          # convert from 0 index to 1 index
          'Measure': measure_num + 1, 
          'BPM': bpm,
          'Stance actions': sas,
          'Previous panels': prev_panels,
        }
        edges_out[prev_node_nm].append(node_nm)
        edges_in[node_nm].append(prev_node_nm)
        prev_node_nm = node_nm

        '''
          Update prev panels
        '''
        for p in active_panel_to_action:
          if p in prev_presses:
            prev_presses.remove(p)
          a = active_panel_to_action[p]
          if a == '1':
            prev_presses.insert(0, p)
          elif a == '2':
            active_holds.add(p)
          elif a == '3':
            active_holds.remove(p)
            prev_presses.insert(0, p)
        prev_presses = prev_presses[:_params.prev_panel_buffer_len[steptype]]
        prev_presses = list(unique_everseen(prev_presses))

        # print(time, bpm, beat, line, len(sas), active_holds, prev_presses)
        # import code; code.interact(local=dict(globals(), **locals()))

      # After processing line, update beat, bpm, and time
      # Important: Update bpm after time
      time_increment = beat_increment * (60 / bpm)
      time += time_increment

      beat += beat_increment
      if beat >= bpms[0][0]:
        # print(beat, bpms)
        bpm = bpms[0][1]
        bpms = bpms[1:]
      assert bpm is not None, 'Failed to set bpm'

    timer.update()

  # Add terminal node and edge
  nodes['final'] = {
    'Time': np.inf,
    'Beat': np.inf,
    'Line': '',
    'Line with active holds': '',
    'Measure': np.inf,
    'BPM': 0,
    'Stance actions': stance.initial_stanceaction(),
    'Previous panels': [],
  }
  edges_out[prev_node_nm].append('final')
  edges_in['final'].append(prev_node_nm)
  edges_out['final'] = []
  edges_out = dict(edges_out)
  edges_in = dict(edges_in)

  return nodes, edges_out, edges_in, stance


def augment_graph_multihits(nodes, edges_out, edges_in, stance, timing_judge = 'piu nj'):
  '''
    Add new nodes and edges when a player can hit multiple notes at different beats with a single hit.
    Augmenting graph after formation helps handle multihits that span measures

    If there are more than 1, hit them in ascending inclusive order. [1, 2, 3] -> [1, 2] and [1, 2, 3]
  '''
  [pre_window, post_window] = _params.perfect_windows[timing_judge]
  num_multihits_proposed = 0
  nms = list(nodes.keys())
  for idx in range(len(nms)):
    nm = nms[idx]
    node = nodes[nm]
    time = node['Time']

    if nm == 'init':
      node['Timing judge'] = timing_judge
      continue

    # Only propose multi hits starting on 1 or 2
    if not has_downpress(node['Line']):
      continue

    multi = []
    for jdx in range(idx + 1, len(nms)):
      if nodes[nms[jdx]]['Time'] - time <= post_window:
        multi.append(nms[jdx])

    if not multi:
      continue

    for jdx in range(len(multi)):
      hits = multi[:jdx + 1]
      last_node_nm = hits[-1]
      last_node = nodes[last_node_nm]
      # Combine line
      lines = [node['Line']] + [nodes[nm]['Line'] for nm in hits]
      joint_line = stance.combine_lines(lines)

      # If num downhits in multihit is the same as the regular hit, skip
      if num_downpress(joint_line) == num_downpress(node['Line']):
        continue

      # Combine line
      aug_lines = [node['Line with active holds']] + [nodes[nm]['Line with active holds'] for nm in hits]
      joint_aug_line = stance.combine_lines(aug_lines)

      sas = stance.get_stanceactions(joint_aug_line, prev_panels = last_node['Previous panels'])

      new_node_nm = f'{nm} multi v{jdx + 1}'
      nodes[new_node_nm] = {
        'Time': last_node['Time'],
        'Beat': last_node['Beat'],
        'Line': joint_line,
        'Line with active holds': joint_aug_line,
        'Measure': last_node['Measure'],
        'BPM': last_node['BPM'],
        'Stance actions': sas,
        'Previous panels': last_node['Previous panels'],
      }
      edges_out[new_node_nm] = edges_out[last_node_nm]
      for n in edges_out[last_node_nm]:
        edges_in[n].append(new_node_nm)
      edges_in[new_node_nm] = edges_in[nm]
      for n in edges_in[nm]:
        edges_out[n].append(new_node_nm)

      # Inspect proposed multihits
      # for key in ['Time', 'Beat', 'Line', 'Line with active holds', 'Measure', 'BPM']:
        # print(key, nodes[new_node_nm][key])
      # import code; code.interact(local=dict(globals(), **locals()))

      num_multihits_proposed += 1

  print(f'Proposed {num_multihits_proposed} multihit nodes')
  return nodes, edges_out, edges_in


'''
  Helper
'''
def has_downpress(line: str) -> bool:
  return bool('1' in line or '2' in line)


def num_downpress(line: str) -> int:
  return line.count('1') + line.count('2')


def has_notes(line: str) -> bool:
  return bool(set(line) != set(['0']))


def parse_bpm(bpms: str) -> List[List]:
  '''
  '''
  bpm_list = []
  for line in bpms.split(','):
    [beat, bpm] = line.split('=')
    bpm_list.append([float(beat), float(bpm)])
  bpm_list.append([np.inf, 0])
  return bpm_list


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
  # nm = 'Super Fantasy - SHK S19 arcade'
  # nm = 'Super Fantasy - SHK S7 arcade'
  # nm = 'Super Fantasy - SHK S4 arcade'
  # nm = 'Final Audition 2 - BanYa S7 arcade'
  # nm = 'Super Fantasy - SHK S10 arcade'

  # Test: Has multi hits
  # nm = 'Sorceress Elise - YAHPP S23 arcade'

  # Test: has hits during holds
  # nm = '8 6 - DASU S20 arcade'

  # Test: Brackets
  # nm = '1950 - SLAM S23 arcade'

  nm = 'HTTP - Quree S21 arcade'

  timing_judge = 'piu nj'

  nodes, edges_out, edges_in, stance = form_graph(nm)
  # Faster than forming graph. More efficient to just run this for each timing judge
  a_nodes, a_edges_out, a_edges_in = augment_graph_multihits(nodes, edges_out, edges_in, stance, timing_judge = timing_judge)

  print(f'Found {len(a_nodes)} nodes')

  with open(out_dir + f'{nm}.pkl', 'wb') as f:
    pickle.dump((a_nodes, a_edges_out, a_edges_in), f)

  return


if __name__ == '__main__':
  main()