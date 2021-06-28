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
all_notes = pickle.load(open(inp_dir_a + f'notes.pkl', 'rb'))
all_bpms = pickle.load(open(inp_dir_a + f'bpms.pkl', 'rb'))
all_warps = pickle.load(open(inp_dir_a + f'warps.pkl', 'rb'))

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

  warps = parse_warps(all_warps[nm])
  beat_to_lines, beats_to_increments = parse_lines_with_warps(measures, warps)

  nodes['init'] = {
    'Time': time,
    'Beat': beat,
    'BPM': bpm,
    'Steptype': steptype,
    'Timing judge': 'None',
  }
  prev_node_nm = 'init'
  edges_in['init'] = []

  timer = util.Timer(total=len(beat_to_lines))
  for beat, line in beat_to_lines.items():
    if _notelines.has_notes(line):
      # Add active holds into line as 4
      aug_line = _notelines.add_active_holds(line, active_holds, stance.panel_to_idx)

      node_nm = f'{beat}'
      nodes[node_nm] = {
        'Time': time,
        'Beat': beat,
        'Line': line,
        'Line with active holds': aug_line,
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
          if p in active_holds:
            active_holds.remove(p)
          else:
            prev_lines = [f'{b:.3f}'.ljust(8) + line for b, line in beat_to_lines.items() if b <= beat]
            print('Bad hold', beat, line)
            print('\n'.join(prev_lines[-10:]))
            import code; code.interact(local=dict(globals(), **locals()))
            raise Exception('Bad hold')

      # print(time, bpm, beat, line, active_holds)
      # import code; code.interact(local=dict(globals(), **locals()))

    bi = beats_to_increments[beat]
    time, bpm, bpms = update_time(time, beat, bi, bpm, bpms)
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
      # import code; code.interact(local=dict(globals(), **locals()))

      num_multihits_proposed += 1

  print(f'Proposed {num_multihits_proposed} multihit nodes')
  return nodes, edges_out, edges_in



'''
  Warping
'''
def parse_warps(warps):
  warps_list = []
  if warps == '':
    return warps_list
  for line in warps.split(','):
    [beat, num_beats] = line.split('=')
    beat = float(beat)
    num_beats = float(num_beats)
    warps_list.append([round(beat, 3), round(beat + num_beats, 3)])
  return warps_list


def beat_in_any_warp(beat, warps):
  # Round to handle beats like 1/3, 2/3
  # Ending needs to be in warp: Obliteration S17
  # Beginning needs to be in warp: V3 S17
  # But sometimes lines are duplicated: Elvis S15
  # in_warp = lambda beat, warp: warp[0] <= round(beat, 3) < warp[1]
  in_warp = lambda beat, warp: warp[0] < round(beat, 3) < warp[1]
  return any(in_warp(beat, warp) for warp in warps)


def beat_begins_any_warp(beat, warps):
  begins_warp = lambda beat, warp: warp[0] == round(beat, 3)
  return any(begins_warp(beat, warp) for warp in warps)


def parse_lines_with_warps(measures, warps):
  beats_per_measure = 4
  beats_to_lines = {}
  beats_to_increments = {}

  warped_beat, unwarped_beat = 0, 0
  for measure_num, measure in enumerate(measures):
    lines = measure.split('\n')
    lines = [line for line in lines if '//' not in line]
    num_subbeats = len(lines)

    for lidx, line in enumerate(lines):
      beat_increment = beats_per_measure / num_subbeats

      line = _notelines.parse_line(line)
      if any(x not in set(list('01234')) for x in line):
        print(f'Error: Bad symbol found in line, {line}')
        raise ValueError(f'Bad symbol found in line, {line}')
      
      if not beat_in_any_warp(unwarped_beat, warps):
        beats_to_lines[warped_beat] = line
        beats_to_increments[warped_beat] = beat_increment
        if not beat_begins_any_warp(unwarped_beat, warps):
          warped_beat += beat_increment
      else:
        # If hold release occurs within warp, add as new line
        if set(line) == set(list('03')):
          prev_beat = list(beats_to_lines.keys())[-1]
          prev_line = beats_to_lines[prev_beat]
          if prev_line.replace('2', '3') == line:
            warp_release_time = 0.001
            release_beat = prev_beat + warp_release_time
            beats_to_lines[release_beat] = line
            beats_to_increments[prev_beat] = warp_release_time
            beats_to_increments[release_beat] = beat_increment - warp_release_time

      unwarped_beat += beat_increment

  # Filter repeated hold releases from warping
  # This occurs from visual gimmicks where holds advance instantly, but do not completely disappear
  beats_to_lines, beats_to_increments = filter_repeated_hold_releases(beats_to_lines, beats_to_increments)
  return beats_to_lines, beats_to_increments


def filter_repeated_hold_releases(beats_to_lines, beats_to_incs):
  '''
    Ignoring empty lines, filter out duplicated hold release lines
    These can only arise from inserting hold releases during warps
  '''
  nonempty_btol = {k: v for k, v in beats_to_lines.items() if set(v) != set('0')}
  beats = list(nonempty_btol.keys())
  assert beats == sorted(beats), 'ERROR: Beats are not sorted by default'
  ok_beats = []
  lines = list(nonempty_btol.values())
  for i in range(len(lines) - 1):
    line1, line2 = lines[i], lines[i+1]
    if set(line1) == set(list('03')) and line1 == line2:
      pass
    else:
      ok_beats.append(beats[i])

  filt_beats_to_lines = {k: v for k, v in beats_to_lines.items() if k in ok_beats}
  filt_beats_to_incs = {k: v for k, v in beats_to_incs.items() if k in ok_beats}
  return filt_beats_to_lines, filt_beats_to_incs


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
  # Update time. No need to update beat, bpm.
  if beat < next_note_beat:
    bi = next_note_beat - beat
    time += bi * (60 / bpm)
  assert bpm is not None, 'ERROR: Failed to set bpm'
  # print(beat, bpm)
  return time, bpm, bpms  


def parse_bpm(bpms):
  '''
  '''
  bpm_list = []
  for line in bpms.split(','):
    [beat, bpm] = line.split('=')
    if float(bpm) < 999:
      # Ignore bpm speedups for visual gimmicks
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
  # nm = 'London Bridge - SCI Guyz S11 arcade'
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
  # nm = 'Obelisque - ESTi x M2U S17 arcade'
  # nm = 'I Want U - MAX S19 arcade'
  # nm = 'Forgotten Vampire - WyvernP S18 arcade'
  # nm = 'The Little Prince (Prod. Godic) - HAON, PULLIK S18 arcade'
  # nm = 'HYPERCUBE - MAX S15 arcade'
  # nm = 'Prime Time - Cashew S23 remix'
  # nm = 'Uranium - Memme S19 arcade'
  # nm = 'Setsuna Trip - Last Note. S16 arcade'
  # nm = 'Gothique Resonance - P4Koo S20 arcade'
  # nm = 'CROSS SOUL - HyuN feat. Syepias S8 arcade'
  # nm = 'CARMEN BUS - StaticSphere & FUGU SUISAN S12 arcade'
  # nm = 'Mr. Larpus - BanYa S22 arcade'
  # nm = 'Bad End Night - HitoshizukuP x yama S17 arcade'
  # nm = 'King of Sales - Norazo S21 arcade'
  # nm = 'Follow me - SHK S9 arcade'
  # nm = 'Death Moon - SHK S22 shortcut'
  # nm = 'Fresh - Aspektz S14 arcade infinity'
  # nm = 'Log In - SHK S20 arcade'
  # nm = 'Phalanx "RS2018 Edit" - Cranky S22 arcade'
  # nm = 'Chicken Wing - BanYa S7 arcade'

  # Test: Has warps
  # nm = 'Wedding Crashers - SHK S16 arcade'
  # nm = 'Gotta Be You - 2NE1 S15 arcade'
  # nm = 'Cross Time - Brandy S18 arcade'
  # nm = 'God Mode feat. skizzo - Nato S20 arcade'
  # nm = 'Sarabande - MAX S20 arcade'
  # nm = 'Nihilism - Another Ver. - - Nato S21 arcade'
  nm = 'Time for the moon night - GFRIEND S16 arcade'
  # nm = 'Full Moon - Dreamcatcher S22 arcade'
  # nm = 'Log In - SHK S20 arcade'
  # nm = 'Elvis - AOA S15 arcade'
  # nm = 'Obliteration - ATAS S17 arcade'

  # Test: Failures
  # nm = 'V3 - Beautiful Day S17 arcade'

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

  # Doubles
  # nm = 'Mitotsudaira - ETIA. D19 arcade'
  # nm = 'Loki - Lotze D19 arcade'
  # nm = 'Trashy Innocence - Last Note. D16 arcade'
  # nm = '8 6 - DASU D21 arcade'
  # nm = 'Bad End Night - HitoshizukuP x yama D18 arcade'

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