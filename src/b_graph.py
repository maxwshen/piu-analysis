'''
  Parse note lines, propose stance actions,
  propose multihits, and annotate edges between lines.

  In this script, nodes = note lines.
  In other scripts, nodes = stance-actions.
'''
import _config, _data, _stances, util, pickle, _params
import sys, os, re, fnmatch, datetime, subprocess
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from typing import List, Dict, Set, Tuple
from more_itertools import unique_everseen

import _notelines, _qsub

# Default params
inp_dir_a = _config.OUT_PLACE + f'a_format_data/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

# Load data
# TODO - Unify into single data structure
all_notes = pickle.load(open(inp_dir_a + f'notes.pkl', 'rb'))
all_bpms = pickle.load(open(inp_dir_a + f'bpms.pkl', 'rb'))

sc_df = pd.read_csv(inp_dir_a + f'all_stepcharts.csv', index_col=0)

# Stances
stance_store = {
  'singles': _stances.Stances(style='singles'),
  'doubles': _stances.Stances(style='doubles'),
}

log_fn = ''

##
# Functions
##
def form_graph(nm, subset_measures = 0):
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
  stance = stance_store[steptype]

  notes = all_notes[nm]
  measures = [s.strip() for s in notes.split(',')]
  bpms = parse_bpm(all_bpms[nm])
  beats_per_measure = 4

  # Testing
  if subset_measures:
    print(f'WARNING: Subsetting to {subset_measures} measures.')
    measures = measures[:subset_measures]

  beat = 0
  time = 0     # units = seconds
  bpm = None
  active_holds = set()
  nodes = dict()
  edges_out = defaultdict(list)
  edges_in = defaultdict(list)

  bpm, bpms = get_init_bpm(beat, bpms)

  nodes['init'] = {
    'Time': time,
    'Beat': beat,
    'Measure': 0,
    'BPM': bpm,
    'Steptype': steptype,
    'Timing judge': 'None',
  }
  prev_node_nm = 'init'
  edges_in['init'] = []

  timer = util.Timer(total=len(measures))
  for measure_num, measure in enumerate(measures):
    lines = measure.split('\n')
    lines = [line for line in lines if '//' not in line]
    num_subbeats = len(lines)
    note_type = num_subbeats
    for lidx, line in enumerate(lines):
      beat_increment = beats_per_measure / num_subbeats

      line = _notelines.parse_line(line)
      if any(x not in set(list('01234')) for x in line):
        print(f'Error: Bad symbol found in line, {line}')
        raise ValueError(f'Bad symbol found in line, {line}')

      if _notelines.has_notes(line):
        # Add active holds into line as 4
        aug_line = _notelines.add_active_holds(line, active_holds, stance.panel_to_idx)

        node_nm = f'{beat}'
        nodes[node_nm] = {
          'Time': time,
          'Beat': beat,
          'Line': line,
          'Line with active holds': aug_line,
          'Measure': measure_num + 1,   # 0-based to 1-based
          'BPM': bpm,
        }

        # Annotate edges for line
        edges_out[prev_node_nm].append(node_nm)
        edges_in[node_nm].append(prev_node_nm)
        prev_node_nm = node_nm

        active_panel_to_action = stance.line_to_panel_to_action(line)
        for p in active_panel_to_action:
          a = active_panel_to_action[p]
          if a == '2':
            active_holds.add(p)
          if a == '3':
            active_holds.remove(p)

        # print(time, bpm, beat, line, active_holds)
        # import code; code.interact(local=dict(globals(), **locals()))

      time, beat, bpm, bpms = update_time(time, beat, beat_increment, bpm, bpms)

    timer.update()

  # Add terminal node and edges
  nodes['final'] = {
    'Time': np.inf,
    'Beat': np.inf,
    'Line': '',
    'Line with active holds': '',
    'Measure': np.inf,
    'BPM': 0,
  }
  edges_out[prev_node_nm].append('final')
  edges_in['final'].append(prev_node_nm)
  edges_out['final'] = []

  return nodes, dict(edges_out), dict(edges_in), stance


def propose_multihits(nodes, edges_out, edges_in, stance, timing_judge = 'piu nj'):
  '''
    Add new nodes and edges when a player can hit multiple notes at different beats with a single hit.
    Augmenting graph after formation helps handle multihits that span measures

    If there are more than 1, hit them in ascending inclusive order. [1, 2, 3] -> [1, 2] and [1, 2, 3]

    Consider at most 3 additional nodes (for 4 total). More can be proposed for incorrectly annotated stepcharts with very high BPM with notes
  '''
  [pre_window, post_window] = _params.perfect_windows[timing_judge]
  num_multihits_proposed = 0
  nms = list(nodes.keys())
  for idx in range(len(nodes)):
    nm = nms[idx]
    node = nodes[nm]
    time = node['Time']

    if nm == 'init':
      node['Timing judge'] = timing_judge
      continue

    if not _notelines.has_downpress(node['Line']):
      continue

    multi = [n for n in nms[idx+1:] if nodes[n]['Time'] - time <= post_window]
    if not multi:
      continue

    multi = multi[:_params.max_lines_in_multihit - 1]
    for jdx in range(len(multi)):
      hits = multi[:jdx + 1]
      last_node_nm = hits[-1]
      last_node = nodes[last_node_nm]
      # Combine line
      lines = [node['Line']] + [nodes[nm]['Line'] for nm in hits]
      joint_line = stance.combine_lines(lines)

      # If num downhits in multihit is the same as the regular hit, skip
      if _notelines.num_downpress(joint_line) == _notelines.num_downpress(node['Line']):
        continue

      # Combine line
      aug_lines = [node['Line with active holds']] + \
                  [nodes[nm]['Line with active holds'] for nm in hits]
      joint_aug_line = stance.combine_lines(aug_lines)

      new_node_nm = f'{nm} multi v{jdx + 1}'
      nodes[new_node_nm] = {
        'Time': last_node['Time'],
        'Beat': last_node['Beat'],
        'Line': joint_line,
        'Line with active holds': joint_aug_line,
        'Measure': last_node['Measure'],
        'BPM': last_node['BPM'],
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
  BPM, beat, and time logic
'''
def update_time(time, beat, beat_increment, bpm, bpms):
  '''
    After processing line, update beat, bpm, and time
    Important: Update time before bpm.
  '''
  next_bpm_update_beat = bpms[0][0]
  next_note_beat = beat + beat_increment

  while next_bpm_update_beat <= next_note_beat:
    # 1 or more bpm updates before next note line.
    # For each bpm update, update beat, time (using bpm+beat), and bpm.
    bi = next_bpm_update_beat - beat
    time += bi * (60 / bpm)
    beat += bi
    if beat >= bpms[0][0]:
      # print(beat, bpms)
      bpm = bpms[0][1]
      bpms = bpms[1:]
      next_bpm_update_beat = bpms[0][0]
    assert bpm is not None, 'ERROR: Failed to set bpm'

  # No more bpm updates before next note line.
  # Update beat, time. No need to update bpm.
  if beat < next_note_beat:
    bi = next_note_beat - beat
    time += bi * (60 / bpm)
    beat += bi
  assert bpm is not None, 'ERROR: Failed to set bpm'
  # print(beat, bpm)
  return time, beat, bpm, bpms  


def parse_bpm(bpms):
  '''
  '''
  bpm_list = []
  for line in bpms.split(','):
    [beat, bpm] = line.split('=')
    bpm_list.append([float(beat), float(bpm)])
  bpm_list.append([np.inf, 0])
  return bpm_list


def get_init_bpm(beat, bpms):
  # Init bpm at beat = 0
  while beat >= bpms[0][0]:
    bpm = bpms[0][1]
    bpms = bpms[1:]
  if bpm is None:
    print('ERROR: Failed to set bpm')
    sys.exit(1)
  return bpm, bpms


'''
  Logging, IO
'''
def output_log(message):
  print(message)
  with open(log_fn, 'a') as f:
    f.write(message)
  return


def load_data(inp_dir, sc_nm):
  with open(inp_dir + f'{sc_nm}.pkl', 'rb') as f:
    line_nodes, line_edges_out, line_edges_in = pickle.load(f)
  return line_nodes, line_edges_out, line_edges_in


'''
  Run
'''
def run_single(sc_nm):
  timing_judge = 'piu nj'

  print(sc_nm, timing_judge)
  global log_fn
  log_fn = out_dir + f'{sc_nm} {timing_judge}.log'

  nodes, edges_out, edges_in, stance = form_graph(sc_nm)

  # Faster than forming graph.
  # More efficient to just run this for each timing judge
  nodes, edges_out, edges_in = propose_multihits(nodes,
      edges_out, edges_in, stance, timing_judge=timing_judge)

  print(f'Found {len(nodes)} nodes')
  with open(out_dir + f'{sc_nm}.pkl', 'wb') as f:
    pickle.dump((nodes, edges_out, edges_in), f)
  # output_log('Success')
  return


@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  # nm = 'Super Fantasy - SHK S19 arcade'
  # nm = 'Super Fantasy - SHK S7 arcade'
  # nm = 'Super Fantasy - SHK S4 arcade'
  # nm = 'Final Audition 2 - BanYa S7 arcade'
  # nm = 'Super Fantasy - SHK S10 arcade'
  # nm = 'Tepris - Doin S17 arcade'
  # nm = 'Last Rebirth - SHK S15 arcade'
  # nm = 'Dawgs In Da House - CanBlaster (Miami Style) S17 arcade'
  # nm = 'Dabbi Doo - Ni-Ni S2 arcade'
  # nm = 'Boulafacet - Nightmare S22 arcade'
  # nm = 'Elvis - AOA S15 arcade'
  # nm = 'Everybody Got 2 Know - MAX S19 remix'
  # nm = 'Chicken Wing - BanYa S17 arcade'
  # nm = 'Hypnosis - BanYa S18 arcade'
  # nm = 'NoNoNo - Apink S14 arcade'
  # nm = 'Rage of Fire - MAX S16 arcade'
  # nm = 'Conflict - Siromaru + Cranky S17 arcade'
  # nm = 'Obelisque - ESTi x M2U S17 arcade'
  # nm = 'I Want U - MAX S19 arcade'
  nm = 'Forgotten Vampire - WyvernP S18 arcade'

  # Test: Has multi hits
  # nm = 'Sorceress Elise - YAHPP S23 arcade'

  # Test: has hits during holds
  # nm = '8 6 - DASU S20 arcade'

  # Test: Brackets
  # nm = '1950 - SLAM S23 arcade'

  # nm = 'HTTP - Quree S21 arcade'
  # nm = '8 6 - DASU S20 arcade'
  # nm = 'Shub Sothoth - Nato & EXC S25 remix'
  # nm = 'The End of the World ft. Skizzo - MonstDeath S20 arcade'
  # nm = 'Loki - Lotze S21 arcade'
  # nm = 'Native - SHK S20 arcade'
  # nm = 'PARADOXX - NATO & SLAM S26 remix'
  # nm = 'BEMERA - YAHPP S24 remix'
  # nm = 'HEART RABBIT COASTER - nato S23 arcade'
  # nm = 'F(R)IEND - D_AAN S23 arcade'
  # nm = 'Pump me Amadeus - BanYa S11 arcade'
  # nm = 'King of Sales - Norazo S21 arcade'
  # nm = 'Hyperion - M2U S20 shortcut'
  # nm = 'Final Audition Ep. 2-2 - YAHPP S22 arcade'
  # nm = 'Achluoias - D_AAN S24 arcade'
  # nm = 'Awakening - typeMARS S16 arcade'

  # Doubles
  # nm = 'Mitotsudaira - ETIA. D19 arcade'
  # nm = 'Loki - Lotze D19 arcade'
  # nm = 'Trashy Innocence - Last Note. D16 arcade'
  # nm = '8 6 - DASU D21 arcade'
  # nm = 'Bad End Night - HitoshizukuP x yama D18 arcade'

  subset_measures = 0
  # subset_measures = 7

  run_single(nm)
  return


if __name__ == '__main__':
  if len(sys.argv) == 1:
    main()
  else:
    if sys.argv[1] == 'gen_qsubs':
      _qsub.gen_qsubs(NAME, sys.argv[2])
    elif sys.argv[1] == 'run_qsubs':
      _qsub.run_qsubs(
        chart_fnm = sys.argv[2],
        start = sys.argv[3],
        end = sys.argv[4],
        run_single = run_single,
      )
    elif sys.argv[1] == 'run_single':
      run_single(sys.argv[2])