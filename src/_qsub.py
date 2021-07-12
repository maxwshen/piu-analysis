import _config, util, subprocess, os
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
    if NAME == 'c_dijkstra':
      if 'highlevel' in chart_fnm:
        vmem = 'h_vmem=8G'
      else:
        vmem = 'h_vmem=4G'
    else:
      vmem = 'h_vmem=1G'

    qsub_commands.append(f'qsub -j y -V -P regevlab -l h_rt=1:00:00,{vmem} -wd {_config.SRC_DIR} {sh_fn} &')

  # Save commands
  commands_fn = qsubs_dir + '_commands.sh'
  with open(commands_fn, 'w') as f:
    f.write('\n'.join(qsub_commands))
  subprocess.check_output(f'chmod +x {commands_fn}', shell = True)
  
  print(f'Wrote {num_scripts} shell scripts to {qsubs_dir}')
  return


def gen_qsubs_remainder(NAME, chart_fnm, extension):
  # Only gen qsubs for unsubmmited jobs, by 
  print('Generating qsub scripts...')
  qsubs_dir = _config.QSUBS_DIR + NAME + '/'
  util.ensure_dir_exists(qsubs_dir)
  qsub_commands = []

  script_id = NAME.split('_')[0]

  out_dir = _config.OUT_PLACE + NAME + '/'
  fns = set(os.listdir(out_dir))

  qdf = pd.read_csv(inp_dir + f'{chart_fnm}.csv', index_col=0)
  names = qdf['Name (unique)']
  num_per_run = len(names) // MAX_QSUB_PROCESSES
  if num_per_run * MAX_QSUB_PROCESSES < len(names):
    num_per_run += 1

  num_scripts = 0
  for i in range(0, len(qdf), num_per_run):
    out_fn = names[i] + extension
    if out_fn in fns:
      continue

    start = i
    end = start + num_per_run
    command = f'python {NAME}.py run_qsubs {chart_fnm} {start} {end}'

    # Write shell scripts
    sh_fn = qsubs_dir + f'q_{script_id}_{chart_fnm}_{start}_{end}.sh'
    with open(sh_fn, 'w') as f:
      f.write(f'#!/bin/bash\n{command}\n')
    num_scripts += 1

    # Write qsub commands
    if NAME == 'c_dijkstra':
      if 'highlevel' in chart_fnm:
        vmem = 'h_vmem=8G'
      else:
        vmem = 'h_vmem=4G'
    else:
      vmem = 'h_vmem=1G'

    qsub_commands.append(f'qsub -j y -V -P regevlab -l h_rt=1:00:00,{vmem} -wd {_config.SRC_DIR} {sh_fn} &')

  # Save commands
  commands_fn = qsubs_dir + '_commands.sh'
  with open(commands_fn, 'w') as f:
    f.write('\n'.join(qsub_commands))
  subprocess.check_output(f'chmod +x {commands_fn}', shell = True)
  
  print(f'Wrote {num_scripts} shell scripts to {qsubs_dir}')
  return


def run_qsubs(chart_fnm, start, end, run_single):
  qdf = pd.read_csv(_config.DATA_DIR + chart_fnm + '.csv', index_col=0)
  qdfs = qdf.iloc[int(start) : int(end)]
  for i, row in qdfs.iterrows():
    try:
      run_single(row['Name (unique)'])
      print('Success', row['Name (unique)'])
    except:
      print('Failed', row['Name (unique)'])
      pass
  return