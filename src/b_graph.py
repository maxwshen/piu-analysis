'''
  Parse note lines, propose stance actions,
  propose multihits, and annotate edges between lines.

  In this script, nodes = note lines.
  In other scripts, nodes = stance-actions.
'''
import _config, _data, _stances, util, pickle, _params
import sys, os, re, fnmatch, datetime, subprocess, copy
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from typing import List, Dict, Set, Tuple
from fractions import Fraction

import _notelines, _qsub, _graph_edit

# Default params
inp_dir_a = _config.OUT_PLACE + f'a_format_data/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

# Load data
all_notes = pickle.load(open(inp_dir_a + f'notes.pkl', 'rb'))
all_bpms = pickle.load(open(inp_dir_a + f'bpms.pkl', 'rb'))
all_warps = pickle.load(open(inp_dir_a + f'warps.pkl', 'rb'))
all_fakes = pickle.load(open(inp_dir_a + f'fakes.pkl', 'rb'))
all_stops = pickle.load(open(inp_dir_a + f'stops.pkl', 'rb'))
all_delays = pickle.load(open(inp_dir_a + f'delays.pkl', 'rb'))

sc_df = pd.read_csv(inp_dir_a + f'all_stepcharts.csv', index_col=0)

# Stances
stance_store = {
  'singles': _stances.Stances(style='singles'),
  'doubles': _stances.Stances(style='doubles'),
}

# Add lines with hold releases in warps at this time increment (seconds)
WARP_RELEASE_TIME = Fraction(1, 1000)

##
# Functions
##
def form_graph(nm, subset_measures = None):
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

  beat_to_lines, beat_to_increments = get_beat_to_lines(measures)
  beat_to_lines = handle_halfdouble(beat_to_lines)

  fakes = parse_data(all_fakes[nm])
  beat_to_lines = apply_fakes(beat_to_lines, fakes)

  warps = parse_data(all_warps[nm], rounding=3)
  bpms = warp_data(warps, bpms)
  beat_to_lines, beat_to_increments = apply_warps(beat_to_lines, beat_to_increments, warps)

  beat_to_lines, beat_to_increments = filter_repeated_hold_releases(beat_to_lines, beat_to_increments)
  beat_to_lines, beat_to_increments = add_empty_lines(beat_to_lines, beat_to_increments)

  stops = parse_data(all_stops[nm])
  stops = warp_data(warps, stops)
  stops_d = tuples_to_dict(stops)

  delays = parse_data(all_delays[nm])
  delays = warp_data(warps, delays)
  delays_d = tuples_to_dict(delays)

  empty_line = list(beat_to_lines.values())[0]

  nodes['init'] = {
    'Time': time,
    'Beat': beat,
    'BPM': bpm,
    'Line': '',
    'Line with active holds': '',
    'Steptype': steptype,
    'Timing judge': 'None',
  }
  prev_node_nm = 'init'
  edges_in['init'] = []

  timer = util.Timer(total=len(beat_to_lines))
  for beat, line in beat_to_lines.items():
    time += delays_d.get(beat, 0)
    if _notelines.has_notes(line):
      # Add active holds into line as 4
      bad_line = False
      try:
        aug_line = _notelines.add_active_holds(line, active_holds, stance.panel_to_idx)
      except:
        # Error when trying to place 4 on 1
        bad_line = True

      active_panel_to_action = stance.line_to_panel_to_action(line)
      bad_hold_releases = []
      for p in active_panel_to_action:
        a = active_panel_to_action[p]
        if a == '3' and p not in active_holds:
          # Tried to release a hold that didn't exist - remove it
          prev_lines = [f'{b:.3f}'.ljust(8) + line for b, line in beat_to_lines.items() if b <= beat]
          # print('Notice: Caught a bad hold', beat, line)
          # print('\n'.join(prev_lines[-10:]))
          pidx = stance.panel_to_idx[p]
          bad_hold_releases.append(pidx)
          bad_line = True
          # import code; code.interact(local=dict(globals(), **locals()))
          # raise Exception('Bad hold')

      # Tried to release a hold that didn't exist - empty the line
      if bad_hold_releases:
        bad_line = True
        # for k in ['Line', 'Line with active holds']:
        #   fixed_line = list(nodes[node_nm][k])
        #   for pidx in bad_hold_releases:
        #     fixed_line[pidx] = '0'
        #   nodes[node_nm][k] = ''.join(fixed_line)

      node_nm = f'{beat}'
      nodes[node_nm] = {
        'Time': time,
        'Beat': beat,
        'Line': line if not bad_line else empty_line,
        'Line with active holds': aug_line if not bad_line else empty_line,
        'BPM': bpm,
      }

      # Annotate edges for line
      edges_out[prev_node_nm].append(node_nm)
      edges_in[node_nm].append(prev_node_nm)
      prev_node_nm = node_nm

      # Update active holds
      if not bad_line:
        for p in active_panel_to_action:
          a = active_panel_to_action[p]
          if a == '2':
            active_holds.add(p)
          if a == '3':
            if p in active_holds:
              active_holds.remove(p)

    bi = beat_to_increments[beat]
    time, bpm, bpms = update_time(time, beat, bi, bpm, bpms)
    time += stops_d.get(beat, 0)
    timer.update()

  # Add terminal node and edges
  nodes['final'] = {
    'Time': np.inf,
    'Beat': np.inf,
    'Line': '',
    'Line with active holds': '',
    'BPM': 0,
  }
  edges_out[prev_node_nm].append('final')
  edges_in['final'].append(prev_node_nm)
  edges_out['final'] = []

  return nodes, dict(edges_out), dict(edges_in), stance


def propose_multihits(nodes, edges_out, edges_in, stance, post_window):
  '''
    Add new nodes and edges when a player can hit multiple notes at different beats with a single hit.
    Augmenting graph after formation helps handle multihits that span measures

    If there are more than 1, hit them in ascending inclusive order. [1, 2, 3] -> [1, 2] and [1, 2, 3]

    Consider at most 3 additional nodes (for 4 total).
  '''
  num_multihits_proposed = 0
  covered_nms = set()
  nms = list(nodes.keys())
  for idx in range(len(nodes)):
    nm = nms[idx]
    node = nodes[nm]
    time = node['Time']

    if nm == 'init':
      node['Multi window'] = post_window
      continue

    if not _notelines.has_downpress(node['Line']):
      continue

    multi = [n for n in nms[idx+1:] if nodes[n]['Time'] - time <= post_window]
    multi = multi[:_params.max_lines_in_multihit - 1]
    if not multi:
      continue

    # Only propose non-overlapping multihits. Implicitly keep earliest multihit.
    if nm in covered_nms:
      continue

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

      # Do not propose multihits that require hands -- very slow
      # This means we currently cannot handle staggered hands
      sd = _notelines.singlesdoubles(joint_line)
      if sd == 'singles':
        if _notelines.num_downpress(joint_line) >= 4:
          continue
      if sd == 'doubles':
        if _notelines.num_downpress(joint_line) >= 5:
          continue

      # Combine line
      aug_lines = [node['Line with active holds']] + \
                  [nodes[nm]['Line with active holds'] for nm in hits]
      joint_aug_line = stance.combine_lines(aug_lines)

      # Require bracketability
      norm_line = joint_aug_line.replace('2', '1').replace('4', '0').replace('3', '0')
      if norm_line not in _params.bracketable_lines:
        continue

      # Disallow rolling hits that include 1 and 2
      if '1' in joint_aug_line and '2' in joint_aug_line:
        continue

      new_node_nm = f'{nm} multi v{jdx + 1}'
      nodes[new_node_nm] = {
        'Time': last_node['Time'],
        'Beat': last_node['Beat'],
        'Line': joint_line,
        'Line with active holds': joint_aug_line,
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

      num_multihits_proposed += 1

      for cnm in [nm] + multi[:jdx + 1]:
        covered_nms.add(cnm)
    
  print(f'Proposed {num_multihits_proposed} multihit nodes')
  return nodes, edges_out, edges_in


def filter_empty_nodes(nodes, edges_out, edges_in):
  empty_set = set(['0'])
  remove = []
  for node in nodes:
    if set(nodes[node]['Line']) == empty_set:
      parents = edges_in[node]
      children = edges_out[node]
      for parent in parents:
        edges_out[parent] = children
      for child in children:
        edges_in[child] = parents
      remove.append(node)
  
  print(f'Deleted {len(remove)} empty line nodes')
  for node in remove:
    del nodes[node]
  return nodes, edges_out, edges_in


def get_beat_to_lines(measures):
  # Includes empty lines
  beats_per_measure = 4
  beat_to_lines = {}
  beat_to_increments = {}
  beat = 0

  for measure_num, measure in enumerate(measures):
    lines = measure.split('\n')
    lines = [line for line in lines if '//' not in line]
    num_subbeats = len(lines)

    for lidx, line in enumerate(lines):
      beat_increment = Fraction(beats_per_measure, num_subbeats)
      line = _notelines.parse_line(line)

      if any(x not in set(list('01234')) for x in line):
        print(f'Error: Bad symbol found in line, {line}')
        raise ValueError(f'Bad symbol found in line, {line}')
      
      beat_to_lines[float(beat)] = line
      beat_to_increments[float(beat)] = beat_increment
      beat += beat_increment

  return beat_to_lines, beat_to_increments


'''
  Warping
'''
def beat_in_any_warp(beat, warps):
  # Round to handle beats like 1/3, 2/3
  # Ending needs to be in warp: Obliteration S17
  # Beginning needs to be in warp: V3 S17
  # But sometimes lines are duplicated: Elvis S15
  # in_warp = lambda beat, warp: warp[0] <= round(beat, 3) < warp[1]
  in_warp = lambda beat, warp: warp[0] < round(beat, 3) < warp[0] + warp[1]
  return any(in_warp(beat, warp) for warp in warps)


def total_warp_beat(beat, warps):
  tot = 0
  for start, length in warps:
    end = start + length
    if end <= beat:
      tot += end - start
    elif start <= beat <= end:
      tot += beat - start
  return tot


def apply_warps(beat_to_lines, beat_to_incs, warps):
  '''
    Remove beats in warps, except:
    - Keep hold release lines compatible with active holds
    Decide to keep start line or end line of warp
    Shift beats after warps
  '''
  # Remove lines in warps, except hold releases
  beats = list(beat_to_lines.keys())
  new_beat_to_lines = {}
  new_beat_to_incs = {}
  nonwarp_beats = set([b for b in beats if not beat_in_any_warp(b, warps)])
  for beat, line in beat_to_lines.items():
    if beat in nonwarp_beats:
      new_beat_to_lines[beat] = line
      new_beat_to_incs[beat] = beat_to_incs[beat]
    elif '3' in line:
      # Line in warp with hold release
      new_beat_to_lines[beat] = line
      new_beat_to_incs[beat] = beat_to_incs[beat]
  beat_to_lines = new_beat_to_lines
  beat_to_incs = new_beat_to_incs

  # Decide to keep start or end line of warp
  is_empty = lambda line: set(line) == set(['0'])
  warp_to_line = {}
  for warp in warps:
    start, end = warp[0], warp[0] + warp[1]
    start_line = beat_to_lines.get(start, None)
    end_line = beat_to_lines.get(end, None)
    # print(start, end, start_line, end_line)
    
    if start_line and not end_line:
      warp_to_line[start] = start_line
    elif not start_line and end_line:
      warp_to_line[start] = end_line
    elif start_line and end_line:
      if is_empty(start_line):
        warp_to_line[start] = end_line
        # print(f'Replaced {start_line} with {end_line} at warped beat {start}')
      elif is_empty(end_line):
        warp_to_line[start] = start_line
        # print(f'Retained {start_line} over {end_line} at warped beat {start}')
      # both start_line and end_line exist and are not empty
      elif start_line.replace('2', '3') != end_line and \
         start_line.replace('2', '1') != end_line and \
         start_line.replace('3', '0') != end_line:
        warp_to_line[start] = end_line
        # print(f'Replaced {start_line} with {end_line} at warped beat {start}')
      else:
        warp_to_line[start] = start_line
        # print(f'Retained {start_line} over {end_line} at warped beat {start}')
    # print(start, end, start_line, end_line, warp_to_line.get(start, None))

  # Shift beats after warps
  new_beat_to_lines = {}
  new_beat_to_incs = {}
  for beat, line in beat_to_lines.items():
    shift = total_warp_beat(beat, warps)
    if beat in nonwarp_beats:
      shifted_beat = beat - shift
    else:
      # hold release line in warp - add very small beat offset
      shifted_beat = beat - shift + WARP_RELEASE_TIME
      while shifted_beat in new_beat_to_lines:
        shifted_beat += WARP_RELEASE_TIME
      print(f'Found hold release in warp; {shifted_beat}, {line}')
    # if beat starts a warp, use warp_to_line, otherwise default to line
    if shifted_beat not in new_beat_to_lines:
      new_beat_to_lines[shifted_beat] = warp_to_line.get(beat, line)
      new_beat_to_incs[shifted_beat] = beat_to_incs[beat]
  
  sorted_beats = sorted(list(new_beat_to_lines.keys()))
  beat_to_lines = {b: new_beat_to_lines[b] for b in sorted_beats}
  beat_to_incs = {b: new_beat_to_incs[b] for b in sorted_beats}
  return beat_to_lines, beat_to_incs


def filter_repeated_hold_releases(beat_to_lines, beat_to_incs):
  '''
    Ignoring empty lines, filter out duplicated hold release lines
    These can only arise from inserting hold releases during warps
  '''
  nonempty_btol = {k: v for k, v in beat_to_lines.items() if set(v) != set('0')}
  beats = list(nonempty_btol.keys())
  empty_beats = [b for b, v in beat_to_lines.items() if set(v) == set('0')]
  assert beats == sorted(beats), 'ERROR: Beats are not sorted by default'
  ok_beats = []
  lines = [nonempty_btol[beat] for beat in sorted(beats)]
  for i in range(len(lines) - 1):
    # filter first in repeated hold release: filter one in warp, not one after warp
    line1, line2 = lines[i], lines[i+1]
    if set(line1) == set(list('03')) and line1 == line2:
      pass
    else:
      ok_beats.append(beats[i])
  ok_beats.append(beats[len(lines)-1])
  ok_beats += empty_beats

  print(f'Filtered {len(beat_to_lines)-len(ok_beats)} repeated hold releases')
  filt_beat_to_lines = {k: v for k, v in beat_to_lines.items() if k in ok_beats}
  filt_beat_to_incs = {k: v for k, v in beat_to_incs.items() if k in ok_beats}
  return filt_beat_to_lines, filt_beat_to_incs


def add_empty_lines(beat_to_lines, beat_to_incs):
  '''
    Every beat + its increment should be a key in both dicts
  '''
  example_line = list(beat_to_lines.values())[0]
  empty_line = '0'*len(example_line)

  # start at beat 0 
  beat_to_incs[-1] = 1

  add_beat_to_lines = {}
  add_beat_to_incs = {}
  beats = set(beat_to_incs.keys())
  for beat in sorted(beat_to_incs.keys()):
    next_beat = beat + beat_to_incs[beat]
    if next_beat not in beats:
      next_beats = [b for b in beats if b > beat]
      if next_beats:
        min_next_beat = min(next_beats)
        min_inc = min([beat_to_incs[beat], beat_to_incs[min_next_beat]])

        nb = beat + min_inc
        while nb < min_next_beat:
          add_beat_to_lines[nb] = empty_line
          add_beat_to_incs[nb] = min_inc
          nb += min_inc

  beat_to_lines.update(add_beat_to_lines)
  beat_to_incs.update(add_beat_to_incs)

  sorted_beats = sorted(list(beat_to_lines.keys()))
  sorted_beats = [b for b in sorted_beats if b >= 0]
  beat_to_lines = {k: beat_to_lines[k] for k in sorted_beats}
  beat_to_incs = {k: beat_to_incs[k] for k in sorted_beats}
  return beat_to_lines, beat_to_incs


def warp_data(warps, data):
  adj_data = []
  for beat, val in data:
    new_start = beat - total_warp_beat(beat, warps)
    adj_data.append((new_start, val))
  start_beats = [s[0] for s in adj_data]
  if start_beats != sorted(start_beats):
    print(f'Error: Warping data failed - no longer in order')
    import code; code.interact(local=dict(globals(), **locals()))
    raise Exception(f'Error: Warping data failed - no longer in order')
  return adj_data


'''
  Fake note sections
  Individual fake notes are handled in _notelines.parse_line
'''
def apply_fakes(beat_to_lines, fakes):
  '''
    Fake ranges are inclusive: Scorpion King S15
    Fakes do not apply to hold releases - BANG BANG BANG S15
  '''
  example_line = list(beat_to_lines.values())[0]
  empty_line = '0' * len(example_line)

  infake = lambda x, fake: fake[0] <= x <= fake[0] + fake[1]
  num_fakes = 0
  for beat, line in beat_to_lines.items():
    if any(infake(beat, r) for r in fakes):
      if set(line) != set(list('03')):
        num_fakes += 1
        beat_to_lines[beat] = empty_line
      # else:
        # print(f'Kept hold release in fake {beat}, {line}')
  print(f'Filtered {num_fakes} fake lines')
  return beat_to_lines


'''
  General data parsing
'''
def parse_data(data, rounding = None):
  res = []
  if data == '':
    return res
  for line in data.split(','):
    [beat, val] = line.split('=')
    beat, val = float(beat), float(val)
    if rounding:
      beat = round(beat, 3)
      val = round(val, 3)
    res.append([beat, val])
  return res


def tuples_to_dict(tuples):
  return {s[0]: s[1] for s in tuples}


'''
  BPM, beat, and time logic
'''
def update_time(time, beat, beat_increment, bpm, bpms):
  '''
    After processing line, update bpm, and time
    Important: Update time before bpm.
  '''
  next_bpm_update_beat = bpms[0][0]
  next_note_beat = beat + beat_increment

  orig_time = copy.copy(time)

  while next_bpm_update_beat <= next_note_beat:
    # 1 or more bpm updates before next note line.
    # For each bpm update, update beat, time (using bpm+beat), and bpm.
    bi = next_bpm_update_beat - beat
    if bi < 0:
      print('Error: Negative beat increment')
      raise Exception('Error: Time decreased')
    time += bi * (60 / bpm)
    beat += bi
    if beat >= bpms[0][0]:
      # print(beat, bpms)
      bpm = bpms[0][1]
      bpms = bpms[1:]
      next_bpm_update_beat = bpms[0][0]
    assert bpm is not None, 'Error: Failed to set bpm'

  # No more bpm updates before next note line.
  # Update time. No need to update beat, bpm.
  if beat < next_note_beat:
    bi = next_note_beat - beat
    time += bi * (60 / bpm)
  assert bpm is not None, 'Error: Failed to set bpm'
  # print(beat, bpm)
  if time < orig_time:
    print('Error: Time decreased')
    raise Exception('ERROR: Time decreased')
  return time, bpm, bpms  


def parse_bpm(bpms):
  bpm_list = []
  for line in bpms.split(','):
    [beat, bpm] = line.split('=')
    bpm_list.append([float(beat), float(bpm)])
    # if float(bpm) < 999:
      # Ignore bpm speedups for visual gimmicks
      # Helps parse some charts, but causes time desyncs
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


def handle_halfdouble(beat_to_lines):
  '''
    Add 00 to each side of lines
  '''
  example_line = list(beat_to_lines.values())[0]
  if len(example_line) == 6:
    return {k: _notelines.hd_to_fulldouble(v) for k, v in beat_to_lines.items()}
  else:
    return beat_to_lines


def summarize_graph(nm, nodes):
  dd = defaultdict(list)
  cols = ['Time', 'Beat', 'BPM', 'Line', 'Line with active holds']
  for node in nodes:
    for col in cols:
      item = nodes[node].get(col, '')
      if col in ['Line', 'Line with active holds']:
        item = _notelines.excel_refmt(item)
      dd[col].append(item)
  df = pd.DataFrame(dd)
  df.to_csv(out_dir + f'{nm}.csv')
  return


'''
  IO
'''
def save_data(nodes, edges_out, sc_nm):
  with open(out_dir + f'{sc_nm}.pkl', 'wb') as f:
    pickle.dump((nodes, edges_out), f)
  return


def load_data(inp_dir, sc_nm):
  fn = inp_dir + f'{sc_nm}.pkl'
  if not os.path.isfile(fn):
    print(f'ERROR: File not found in b_graph')
    raise Exception(f'ERROR: File not found in b_graph')
  with open(fn, 'rb') as f:
    line_nodes, line_edges_out = pickle.load(f)
  return line_nodes, line_edges_out


'''
  Run
'''
def run_single(sc_nm):
  post_window = _params.multi_window
  # timing_judge = 'piu nj'
  # [pre_window, post_window] = _params.perfect_windows[timing_judge]

  print(sc_nm)

  nodes, edges_out, edges_in, stance = form_graph(sc_nm)
  nodes, edges_out, edges_in = filter_empty_nodes(nodes, edges_out, edges_in)

  # Faster than forming graph. More efficient to just run this for each timing judge
  nodes, edges_out, edges_in = propose_multihits(nodes,
      edges_out, edges_in, stance, post_window)

  # Ensure edges are unique
  for node in edges_out:
    edges_out[node] = list(set(edges_out[node]))
  for node in edges_in:
    edges_in[node] = list(set(edges_in[node]))

  # Remove regular nodes covered by multis - force multi use
  nodes, edges_out = _graph_edit.edit(nodes, edges_out, edges_in)

  # Sort nodes by beat
  beat_to_node = defaultdict(list)
  for node in nodes:
    beat = nodes[node]['Beat']
    beat_to_node[beat].append(node)
  sorted_beats = sorted(list(beat_to_node.keys()))
  reordered_nodes = {}
  for b in sorted_beats:
    for node in beat_to_node[b]:
      reordered_nodes[node] = nodes[node]
  nodes = reordered_nodes

  print(f'Found {len(nodes)} nodes')
  save_data(nodes, edges_out, sc_nm)
  summarize_graph(sc_nm, nodes)
  return


@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  # nm = 'Super Fantasy - SHK S19 arcade'
  # nm = 'Nakakapagpabagabag - Dasu feat. Kagamine Len S18 arcade'
  # nm = 'Phalanx "RS2018 Edit" - Cranky S22 arcade'
  # nm = 'Xeroize - FE S24 arcade'
  # nm = 'Super Fantasy - SHK S7 arcade'
  # nm = 'Super Fantasy - SHK S4 arcade'
  # nm = 'Final Audition 2 - BanYa S7 arcade'
  # nm = 'Super Fantasy - SHK S10 arcade'
  # nm = 'Tepris - Doin S17 arcade'
  # nm = 'Last Rebirth - SHK S15 arcade'
  # nm = 'Dawgs In Da House - CanBlaster (Miami Style) S17 arcade'
  # nm = 'Dabbi Doo - Ni-Ni S2 arcade'
  # nm = 'Boulafacet - Nightmare S22 arcade'
  # nm = 'Everybody Got 2 Know - MAX S19 remix'
  # nm = 'Chicken Wing - BanYa S17 arcade'
  # nm = 'Hypnosis - BanYa S18 arcade'
  # nm = 'NoNoNo - Apink S14 arcade'
  # nm = 'Rage of Fire - MAX S16 arcade'
  # nm = 'Conflict - Siromaru + Cranky S17 arcade'
  # nm = 'Final Audition - BanYa S15 arcade'
  # nm = 'Oy Oy Oy - BanYa S13 arcade'
  # nm = 'An Interesting View - BanYa S13 arcade'
  # nm = 'Bee - BanYa S15 arcade'
  # nm = 'Beat of The War 2 - BanYa S21 arcade'
  # nm = 'Exceed2 Opening - Banya S15 shortcut'
  # nm = 'The Little Prince (Prod. Godic) - HAON, PULLIK S9 arcade'
  # nm = 'The Little Prince (Prod. Godic) - HAON, PULLIK S13 arcade'
  # nm = 'Obelisque - ESTi x M2U S17 arcade'
  # nm = 'I Want U - MAX S19 arcade'
  # nm = 'Forgotten Vampire - WyvernP S18 arcade'
  # nm = 'The Little Prince (Prod. Godic) - HAON, PULLIK S18 arcade'
  # nm = 'HYPERCUBE - MAX S15 arcade'
  # nm = 'Prime Time - Cashew S23 remix'
  # nm = 'Uranium - Memme S19 arcade'
  # nm = 'Setsuna Trip - Last Note. S16 arcade'
  # nm = 'CROSS SOUL - HyuN feat. Syepias S8 arcade'
  # nm = 'CARMEN BUS - StaticSphere & FUGU SUISAN S12 arcade'
  # nm = 'Mr. Larpus - BanYa S22 arcade'
  # nm = 'Bad End Night - HitoshizukuP x yama S17 arcade'
  # nm = 'King of Sales - Norazo S21 arcade'
  # nm = 'Follow me - SHK S9 arcade'
  # nm = 'Death Moon - SHK S22 shortcut'
  # nm = 'Fresh - Aspektz S14 arcade infinity'
  # nm = 'Phalanx "RS2018 Edit" - Cranky S22 arcade'
  # nm = 'Chicken Wing - BanYa S7 arcade'
  # nm = 'Super Fantasy - SHK S16 arcade'
  # nm = 'Bad Apple!! feat. Nomico - Masayoshi Minoshima S17 arcade'

  # Test: Has warps
  # nm = 'Wedding Crashers - SHK S16 arcade'
  # nm = 'Gotta Be You - 2NE1 S15 arcade'
  # nm = 'Cross Time - Brandy S18 arcade'
  # nm = 'God Mode feat. skizzo - Nato S20 arcade'
  # nm = 'Sarabande - MAX S20 arcade'
  # nm = 'Nihilism - Another Ver. - - Nato S21 arcade'
  # nm = 'Time for the moon night - GFRIEND S16 arcade'
  # nm = 'Poseidon - Quree S20 arcade'
  # nm = 'Tales of Pumpnia - Applesoda S16 arcade'
  # nm = 'Acquaintance - Outsider S17 arcade'
  # nm = 'Full Moon - Dreamcatcher S22 arcade'
  # nm = 'Elvis - AOA S15 arcade'
  # nm = 'Gothique Resonance - P4Koo S20 arcade'
  # nm = 'Shub Niggurath - Nato S24 arcade'
  # nm = 'Club Night - Matduke S18 arcade'
  # nm = 'Obliteration - ATAS S17 arcade'

  # Test: Fake notes
  # nm = 'Club Night - Matduke S18 arcade'
  # nm = 'Good Night - Dreamcatcher S20 arcade'
  # nm = 'God Mode feat. skizzo - Nato S18 arcade'

  # Test: Failures
  # nm = 'PRIME - Tatsh S7 arcade'
  # nm = '%X (Percent X) - Pory S17 arcade'
  # nm = 'Log In - SHK S20 arcade'
  # nm = 'Shub Niggurath - Nato S24 arcade'
  # nm = 'Allegro Con Fuoco - FULL SONG - - DM Ashura S23 fullsong'
  # nm = 'Club Night - Matduke D21 arcade'
  # nm = 'Macaron Day - HyuN D18 arcade'
  # nm = 'V3 - Beautiful Day S17 arcade'
  # nm = 'Death Moon - SHK S17 arcade'
  # nm = 'Tales of Pumpnia - Applesoda S16 arcade'
  # nm = 'Cowgirl - Bambee HD11 arcade'
  # nm = 'Chicken Wing - BanYa HD16 arcade'
  # nm = 'Wedding Crashers - SHORT CUT - - SHK S18 shortcut'
  # nm = 'Desaparecer - Applessoda vs MAX S20 remix'
  # nm = '1950 - SLAM DP3 arcade'
  # nm = "Rave 'til the Earth's End - 5argon S17 arcade"
  # nm = 'Sarabande - MAX S20 arcade'
  # nm = 'Leather - Doin D22 remix'
  # nm = 'You Got Me Crazy - MAX D18 arcade'
  # nm = 'Accident - MAX S18 arcade'
  # nm = 'Scorpion King - r300k S15 arcade'
  # nm = 'Red Swan - Yahpp S18 arcade'
  # nm = 'Requiem - MAX D23 arcade'
  # nm = 'Good Night - Dreamcatcher S17 arcade'
  # nm = 'Fly high - Dreamcatcher S15 arcade'
  # nm = 'Poseidon - Quree S20 arcade'
  # nm = 'HANN (Alone) - (G)I-DLE D17 arcade'
  # nm = 'BANG BANG BANG - BIGBANG S15 arcade'
  # nm = 'PRIME - Tatsh S11 arcade'
  # nm = 'Wedding Crashers - SHORT CUT - - SHK S18 shortcut'

  # Test: Many hands
  # nm = 'London Bridge - SCI Guyz S11 arcade'

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
  # nm = 'Hyperion - M2U S20 shortcut'
  # nm = 'Final Audition Ep. 2-2 - YAHPP S22 arcade'
  # nm = 'Achluoias - D_AAN S24 arcade'
  # nm = 'Awakening - typeMARS S16 arcade'
  # nm = 'Love is a Danger Zone - BanYa S7 arcade'
  # nm = 'Scorpion King - r300k S8 arcade'
  # nm = 'Imagination - SHK S17 arcade'
  nm = 'Enhanced Reality - Matduke S16 arcade'

  # Doubles
  # nm = 'Mitotsudaira - ETIA. D19 arcade'
  # nm = 'Headless Chicken - r300k D21 arcade'
  # nm = 'King of Sales - Norazo D19 arcade'
  # nm = 'Canon D - BanYa D17 arcade'
  # nm = 'Shock - BEAST D15 arcade'
  # nm = 'Witch Doctor #1 - YAHPP HD19 arcade'
  # nm = 'Slam - Novasonic D19 arcade'
  # nm = 'Emperor - BanYa D17 arcade'
  # nm = 'Loki - Lotze D19 arcade'
  # nm = 'Trashy Innocence - Last Note. D16 arcade'
  # nm = '8 6 - DASU D21 arcade'
  # nm = 'Bad End Night - HitoshizukuP x yama D18 arcade'
  # nm = 'Maslo - Vospi D16 arcade'
  # nm = 'Energetic - Wanna One D19 arcade'
  # nm = 'You Got Me Crazy - MAX D18 arcade'
  # nm = 'Anguished Unmaking - void D18 arcade'
  # nm = 'Poseidon - SHORT CUT - - Quree D14 shortcut'
  # nm = 'Ugly Dee - Banya Production D15 arcade'
  # nm = 'Destination - SHK D19 shortcut'
  # nm = 'JANUS - MAX D14 arcade'
  # nm = 'PICK ME - PRODUCE 101 DP3 arcade'
  # nm = 'She Likes Pizza - BanYa D16 arcade'
  # nm = 'Break Out - Lunatic Sounds D22 arcade'
  # nm = 'Mr. Larpus - BanYa D14 arcade'
  # nm = 'Windmill - Yak Won D23 arcade'
  # nm = 'Indestructible - Matduke D22 arcade'
  # nm = 'Rock the house - Matduke D22 arcade'
  # nm = 'Round and Round - Eskimo & Icebird D12 arcade'
  # nm = 'Ugly Dee - Banya Production D15 arcade'
  # nm = 'Conflict - Siromaru + Cranky D24 arcade'

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
    elif sys.argv[1] == 'gen_qsubs_remainder':
      _qsub.gen_qsubs_remainder(NAME, sys.argv[2], '.pkl')
    elif sys.argv[1] == 'run_single':
      run_single(sys.argv[2])