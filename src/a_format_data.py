# 
import _config, _data, util, pickle
import sys, os, fnmatch, datetime, subprocess
import numpy as np
from collections import defaultdict, Counter
import pandas as pd

# Default params
inp_dir = _config.DATA_DIR
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

##
# Functions
##
def get_all_stepcharts_df() -> None:
  '''
    Loads and filters stepcharts from .ssc files
    Outputs
    - all_stepcharts.csv: CSV describing each stepchart
    - notes.pkl: Dict mapping stepchart name to notes string
  '''
  df = _data.datasets['all']
  print(f'Importing {len(df)} .ssc files ...')
  mdf = pd.DataFrame()
  all_notes = []
  all_bpms = []
  timer = util.Timer(total=len(df))
  for idx, row in df.iterrows():
    ssc_fn = row['Files']
    sc = _data.SSCFile(ssc_fn, pack=row['Pack'])

    sc_df = sc.get_stepchart_info()
    mdf = mdf.append(sc_df, ignore_index=True)

    # Same order as mdf
    sc_notes = sc.get_stepchart_notes()
    all_notes += sc_notes

    bpms = sc.get_bpms()
    all_bpms += bpms

    timer.update()

  mdf['Stepchart index'] = mdf.index
  print(f'Found {len(mdf)} total stepcharts ...')

  # Filter down to singles and doubles only
  ok_steptypes = ['pump-single', 'pump-double', 'pump-halfdouble']
  print(f'Filtering down to {ok_steptypes}...')
  crit = (mdf['STEPSTYPE'].isin(ok_steptypes))
  mdf = mdf[crit].reset_index(drop=True)
  print(f'... retained {len(mdf)} stepcharts ...')

  # Filter UCS out
  print(f'Filtering down to non-UCS...')
  crit = (mdf['Is UCS'] == False)
  mdf = mdf[crit].reset_index(drop=True)
  print(f'... retained {len(mdf)} stepcharts ...')

  # Filter to standard packs -- some repeats in basic mode
  print(f'Filtering down to standard packs...')
  crit = (mdf['Pack'].isin(_data.standard_packs))
  mdf = mdf[crit].reset_index(drop=True)
  print(f'... retained {len(mdf)} stepcharts ...')

  # Filter quest
  print(f'Filtering quest stepcharts...')
  crit = (mdf['Is quest'] == False)
  mdf = mdf[crit].reset_index(drop=True)
  print(f'... retained {len(mdf)} stepcharts ...')

  # Filter special
  print(f'Filtering special stepcharts...')
  crit = (mdf['SONGTYPE'] != 'SPECIAL')
  mdf = mdf[crit].reset_index(drop=True)
  print(f'... retained {len(mdf)} stepcharts ...')

  # Filter time signatures other than 4/4
  print(f'Filtering time signatures other than 4/4 ...')
  bad_sigs = [s for s in mdf['TIMESIGNATURES'] if ',' in s]
  bad_sigs += [s for s in mdf['TIMESIGNATURES'] if s[-3:] != '4=4']
  bad_sigs = set(bad_sigs)
  crit = (~mdf['TIMESIGNATURES'].isin(bad_sigs))
  mdf = mdf[crit].reset_index(drop=True)
  print(f'... retained {len(mdf)} stepcharts ...')

  # Look at non-unique names
  nmcts = Counter(mdf['Name'])
  dup_nms = sorted(nmcts, key = nmcts.get, reverse=True)
  dup_nms = [s for s in dup_nms if nmcts[s] > 1]
  # res = [(s, nmcts[s]) for s in dup_nms]

  # Make names unique
  num_unique_nms = len(set(mdf['Name']))
  print(f'Found {num_unique_nms} unique names')

  print(f'Making names unique ...')
  seen = defaultdict(lambda: 0)
  dup_nms = set(dup_nms)
  nms = list(mdf['Name'])
  for idx, nm in enumerate(nms):
    if nm in dup_nms:
      seen[nm] += 1
      nms[idx] = f'{nm} v{seen[nm]}'
  mdf['Name'] = nms
  mdf = mdf.rename(columns = {'Name': 'Name (unique)'})

  num_unique_nms = len(set(mdf['Name (unique)']))
  print(f'Found {num_unique_nms} unique names')
  assert num_unique_nms == len(mdf)

  # Save
  print(f'Total: {len(mdf)} stepcharts ...')
  mdf.to_csv(out_dir + f'all_stepcharts.csv')

  # Handle notes -- filter, then convert to dict
  all_notes_d = {}
  print(f'Filtering notes ...')
  timer = util.Timer(total = len(mdf))
  for idx, row in mdf.iterrows():
    notes = all_notes[row['Stepchart index']]
    all_notes_d[row['Name (unique)']] = notes
    timer.update()

  with open(out_dir + f'notes.pkl', 'wb') as f:
    pickle.dump(all_notes_d, f)

  # Handle bpms -- filter, then convert to dict
  all_bpms_d = {}
  print(f'Filtering bpms ...')
  timer = util.Timer(total = len(mdf))
  for idx, row in mdf.iterrows():
    bpms = all_bpms[row['Stepchart index']]
    all_bpms_d[row['Name (unique)']] = bpms
    timer.update()

  with open(out_dir + f'bpms.pkl', 'wb') as f:
    pickle.dump(all_bpms_d, f)

  return


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
    command = f'python {NAME}.py {idx}'
    script_id = NAME.split('_')[0]

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{idx}.sh'
    with open(sh_fn, 'w') as f:
      f.write(f'#!/bin/bash\n{command}\n')
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append(f'qsub -j y -V -wd {_config.SRC_DIR} {sh_fn}')

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print(f'Wrote {num_scripts} shell scripts to {qsubs_dir}')
  return

##
# Main
##
@util.time_dec
def main():
  print(NAME)
  
  get_all_stepcharts_df()

  # test_ssc_fn = '/mnt/c/Users/maxws/Downloads/StepF2/Songs/19-XX/(00) 1608 - MAX - I Want U/1608 - MAX - I Want U.ssc'
  # sc = _data.SSCFile(test_ssc_fn)

  # # import_all_sscs()

  # format_data(ssc_fn=_config.SRC_DIR + 'timeforthemoonnight-s18.ssc')
  return


if __name__ == '__main__':
  main()