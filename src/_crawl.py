import _config
import _data
from collections import defaultdict
import numpy as np, pandas as pd
import os, re, itertools
from typing import List, Dict, Set, Tuple

STEP_FOLD = '/mnt/c/Users/maxws/Downloads/StepP1/Songs/'

'''
  Crawling
'''
def get_folder_items_paths(fold):
  return [os.path.join(fold, s) for s in os.listdir(fold)]


def get_sub_folds(fold):
  return filter(os.path.isdir, get_folder_items_paths(fold))


def get_ssc(fold):
  return filter(lambda s: re.match('.*\.ssc', s), get_folder_items_paths(fold))


'''
  Parse
'''
def get_pack_name(fold):
  '''
    /STEP_FOLD/18-PRIME 2/
    -> 'PRIME 2'
  '''
  return fold.split('/')[-1].split('-')[-1]


'''
  Primary
'''
def crawl_all_ssc():
  '''
    Find all .ssc files in child folders of STEP_FOLD
    /STEP_FOLD/18-PRIME 2/15A2 - Start on Red/15A2 - Start On Red - Nato.ssc

    Assumptions:
    - input `STEP_FOLD`
    - `STEP_FOLD` contains `pack` folders, e.g., `PRIME 2`
    - .ssc files are exactly 2 subdirectories down
  '''
  print(f'Crawling local .sscs in {STEP_FOLD} ...')

  sfolds = get_sub_folds(STEP_FOLD)
  dd = defaultdict(list)
  for sfold in sfolds:
    print(sfold,)
    pack_nm = get_pack_name(sfold)
    ssfolds = get_sub_folds(sfold)

    ssc_fns = list(itertools.chain(*(get_ssc(sf) for sf in ssfolds)))

    dd['Pack'] += [pack_nm] * len(ssc_fns)
    dd['Files'] += ssc_fns
    print(f'\tFound {len(ssc_fns)} .sscs')

  df = pd.DataFrame(dd)
  df.to_csv(_config.DATA_DIR + f'local_stepf2_files.csv')
  return df


if __name__ == '__main__':
  crawl_all_ssc()