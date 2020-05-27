# 
from __future__ import division
import _config, _data, util
import sys, os, fnmatch, datetime, subprocess
import numpy as np
from collections import defaultdict
import pandas as pd

# Default params
inp_dir = _config.DATA_DIR
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

##
# Functions
##
def format_data(ssc_fn):

  return


def import_all_sscs():
  df = _data.datasets['all']
  print(f'Importing {len(df)} .ssc files ...')
  timer = util.Timer(total = len(df))
  for idx, row in df.iterrows():
    ssc_fn = row['Files']
    sc = _data.SSCFile(ssc_fn)
    timer.update()
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
  
  # _data.crawl_all_ssc()

  test_ssc_fn = '/mnt/c/Users/maxws/Downloads/StepF2/Songs/19-XX/(00) 1608 - MAX - I Want U/1608 - MAX - I Want U.ssc'
  sc = _data.SSCFile(test_ssc_fn)

  # import_all_sscs()

  format_data(
    ssc_fn = _config.SRC_DIR + 'timeforthemoonnight-s18.ssc'
  )

  return


if __name__ == '__main__':
  main()