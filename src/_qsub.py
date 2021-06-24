import _config, util
import pandas as pd

inp_dir = _config.DATA_DIR

MAX_QSUB_PROCESSES = 950

def gen_qsubs(NAME, chart_fnm):
  # Generate qsub shell scripts and commands for easy parallelization
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  script_id = NAME.split('_')[0]

  qdf = pd.read_csv(inp_dir + chart_fnm + '.csv', index_col=0)
  names = qdf['Name (unique)']
  num_per_run = len(names) // MAX_QSUB_PROCESSES
  if num_per_run * MAX_QSUB_PROCESSES < len(names):
    num_per_run += 1

  num_scripts = 0
  for start in range(0, len(qdf), num_per_run):
    end = start + num_per_run
    command = f'python {NAME}.py run_qsubs {chart_fnm} {start} {end}'

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{chart_fnm}_{start}_{end}.sh'
    with open(sh_fn, 'w') as f:
      f.write(f'#!/bin/bash\n{command}\n')
    num_scripts += 1

    # Write qsub commands
    qsub_commands.append(f'qsub -j y -V -P regevlab -wd {_config.SRC_DIR} {sh_fn}')

  # Save commands
  with open(qsubs_dir + '_commands.txt', 'w') as f:
    f.write('\n'.join(qsub_commands))

  print(f'Wrote {num_scripts} shell scripts to {qsubs_dir}')
  return


def run_qsubs(chart_fnm, start, end, run_single):
  qdf = pd.read_csv(_config.DATA_DIR + chart_fnm + '.csv', index_col=0)
  qdfs = qdf.iloc[int(start) : int(end)]
  for i, row in qdfs.iterrows():
    run_single(row['Name (unique)'])
  return