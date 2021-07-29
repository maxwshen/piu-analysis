'''
  Merge features
  also predict difficulty
'''

import _config, util
import sys, os, subprocess
import pandas as pd

import predict_difficulty

inp_dir = _config.OUT_PLACE + 'd_annotate/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

fns = [fn for fn in os.listdir(inp_dir) if '_features' in fn]
num_jobs = 20

all_df = pd.read_csv(_config.OUT_PLACE + 'a_format_data/all_stepcharts.csv', index_col=0)
all_df['Level'] = all_df['METER']
all_df['Is singles'] = (all_df['Steptype simple'].str.contains('S'))
all_df['Is doubles'] = (all_df['Steptype simple'].str.contains('D'))


def merge(start, end):
  start, end = int(start), int(end)

  sub_fns = fns[start:end]

  timer = util.Timer(total=len(sub_fns))
  mdf = pd.DataFrame()
  for fn in sub_fns:
    df = pd.read_csv(inp_dir + fn, index_col=0)
    mdf = mdf.append(df)
    timer.update()
  
  mdf.to_csv(out_dir + f'features-{start}-{end}.csv')
  return


def merge_all():
  mdf = pd.DataFrame()
  num_per_run = get_num_per_run()
  timer = util.Timer(total=num_jobs)
  print('Merging ...')
  for start in range(0, len(fns), num_per_run):
    end = start + num_per_run
    df = pd.read_csv(out_dir + f'features-{start}-{end}.csv', index_col=0)
    mdf = mdf.append(df)
    timer.update()
  
  # exclude chart info from ML feature columns
  ft_cols = [x for x in mdf.columns if x != 'Name (unique)']
  mdf = mdf.merge(all_df, on='Name (unique)', how='left')

  print('Training models to predict difficulty ...')
  dfs = predict_difficulty.predict(mdf[mdf['Is singles']], ft_cols)
  dfd = predict_difficulty.predict(mdf[mdf['Is doubles']], ft_cols)

  mdf = dfs.append(dfd)
  mdf.to_csv(out_dir + 'features.csv')
  return


def get_num_per_run():
  num_per_run = len(fns) // num_jobs
  if num_per_run * num_jobs < len(fns):
    num_per_run += 1
  return num_per_run

#
def gen_qsubs():
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  script_id = NAME.split('_')[0]

  num_per_run = get_num_per_run()

  num_scripts = 0
  for start in range(0, len(fns), num_per_run):
    end = start + num_per_run
    command = f'python {NAME}.py run_single {start} {end}'

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{start}_{end}.sh'
    with open(sh_fn, 'w') as f:
      f.write(f'#!/bin/bash\n{command}\n')
    num_scripts += 1

    qsub_commands.append(f'qsub -j y -V -P regevlab -l h_rt=1:00:00, -wd {_config.SRC_DIR} {sh_fn} &')

  # Save commands
  commands_fn = qsubs_dir + '_commands.sh'
  with open(commands_fn, 'w') as f:
    f.write('\n'.join(qsub_commands))
  subprocess.check_output(f'chmod +x {commands_fn}', shell = True)
  
  print(f'Wrote {num_scripts} shell scripts to {qsubs_dir}')
  return


if __name__ == '__main__':
  if len(sys.argv) == 1:
    print('-- Intended to run with qsub')
  else:
    if sys.argv[1] == 'gen_qsubs':
      gen_qsubs()
    elif sys.argv[1] == 'run_single':
      merge(
        start = sys.argv[2],
        end = sys.argv[3]
      )
    elif sys.argv[1] == 'merge_all':
      merge_all()