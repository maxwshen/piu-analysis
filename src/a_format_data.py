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
def get_all_stepcharts_df():
  '''
    Loads and filters stepcharts from .ssc files
    Outputs
    - all_stepcharts.csv: CSV describing each stepchart
    - notes.pkl: Dict mapping stepchart name to notes string
  '''
  df = _data.datasets['all']
  print(f'Importing {len(df)} .ssc files ...')
  mdf = pd.DataFrame()
  all_attributes = defaultdict(list)
  timer = util.Timer(total=len(df))
  for idx, row in df.iterrows():
    ssc_fn = row['Files']
    sc = _data.SSCFile(ssc_fn, pack=row['Pack'])

    sc_df = sc.get_stepchart_info()
    mdf = mdf.append(sc_df, ignore_index=True)

    # Same order as mdf. Keys will be output file: f'{key}.pkl'
    all_attributes['notes'] += sc.get_stepchart_notes()
    all_attributes['bpms'] += sc.get_attribute('BPMS')
    all_attributes['tickcounts'] += sc.get_attribute('TICKCOUNTS')
    all_attributes['warps'] += sc.get_attribute('WARPS')
    all_attributes['fakes'] += sc.get_attribute('FAKES')

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
  nms = [nm.replace('/', '') for nm in nms]
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

  for name, all_d in all_attributes.items():
    filter_and_save(name, all_d, mdf)
  return


def filter_and_save(name, all_data, mdf):
  # Filter, then convert to dict
  all_d = {}
  print(f'Filtering {name} ...')
  timer = util.Timer(total = len(mdf))
  for idx, row in mdf.iterrows():
    all_d[row['Name (unique)']] = all_data[row['Stepchart index']]
    timer.update()
  with open(out_dir + f'{name}.pkl', 'wb') as f:
    pickle.dump(all_d, f)
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