# 
from __future__ import division
import _config, _data, _stances, util, pickle
import sys, os, fnmatch, datetime, subprocess
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from typing import List, Dict, Set, Tuple

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
  'singles': _stances.SinglesStances(),
  'doubles': _stances.DoublesStances(),
}

##
# Functions
##
def form_graph(nm: str) -> None:
  '''
    Assumes 4/4 time signature (other time sigs are extremely rare, and assumed to be filtered out)

    Notes format: https://github.com/stepmania/stepmania/wiki/sm.

    TODO -- decide on "outputs" of this function
    - Want to store annotations of note types and bpms for runs, probably in pd.DataFrame?
    - Also need to propose and store graph
  '''
  atts = sc_df[sc_df['Name (unique)'] == nm]
  if 'S' in atts['Steptype simple']:
    stance = stance_store['singles']
  elif 'D' in atts['Steptype simple']:
    stance = stance_store['doubles']
  else:
    assert False, 'No stance found' 

  notes = all_notes[nm]
  measures = [s.strip() for s in notes.split(',')]
  bpms = parse_bpm(all_bpms[nm])
  beats_per_measure = 4

  beat = 0
  time = 0     # units = seconds
  bpm = None
  holds = dict()
  timer = util.Timer(total = len(measures))
  for measure in measures:
    lines = measure.split('\n')
    num_subbeats = len(lines)
    note_type = num_subbeats
    for line in lines:
      beat_increment = beats_per_measure / num_subbeats

      # Process
      if has_notes(line):
        # TODO: Form constraints using previous holds and current line
        nodes = stance.get_stanceactions(line)

        print(time, bpm, beat)
        import code; code.interact(local=dict(globals(), **locals()))

      # After processing line, update beat, bpm, and time
      beat += beat_increment
      if beat >= bpms[0][0]:
        bpm = bpms[0][1]
        bpms = bpms[1:]
      assert bpm is not None, 'Failed to set bpm'

      time_increment = beat_increment * (60 / bpm)
      time += time_increment
  timer.update()

  return


def has_notes(line: str) -> bool:
  return bool(set(line) != set(['0']))


def parse_bpm(bpms: str) -> List[List]:
  '''
  '''
  bpm_list = []
  for line in bpms.split(','):
    [beat, bpm] = line.split('=')
    bpm_list.append([float(beat), float(bpm)])
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
    qsub_commands.append('qsub -V -wd %s %s' % (_config.SRC_DIR, sh_fn))

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
  form_graph(nm)

  return


if __name__ == '__main__':
  main()