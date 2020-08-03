import _config
import _data
from collections import defaultdict
import numpy as np, pandas as pd
import os
from typing import List, Dict, Set, Tuple

step_fold = '/mnt/c/Users/maxws/Downloads/StepP1/Songs/'

'''
  Crawling
'''
def crawl_all_ssc():
  '''
    Find all .ssc files in child folders of step_fold
    /step_fold/18-PRIME 2/15A2 - Start on Red/15A2 - Start On Red - Nato.ssc
  '''
  print(f'Crawling local .sscs in {step_fold} ...')
  get_sub_folds = lambda f: [os.path.join(f, s) for s in os.listdir(f) if os.path.isdir(os.path.join(f, s))]
  ssc_matcher = '.ssc'
  get_ssc = lambda f: [os.path.join(f, s) for s in os.listdir(f) if s[-len(ssc_matcher):] == ssc_matcher]

  sfolds = get_sub_folds(step_fold)
  dd = defaultdict(list)
  for sfold in sfolds:
    print(sfold,)
    pack_nm = sfold.split('/')[-1].split('-')[-1]
    ssfolds = get_sub_folds(sfold)

    fns = []
    for ssfold in ssfolds:
      sscs = get_ssc(ssfold)
      fns += sscs

    dd['Pack'] += [pack_nm] * len(fns)
    dd['Files'] += fns
    dd['Song name'] += [s.split('/')[-2].split(' - ')[-1] for s in fns]
    print(f'\tFound {len(fns)} .sscs')

  df = pd.DataFrame(dd)
  df.to_csv(_config.DATA_DIR + f'local_stepf2_files.csv')
  return df


if __name__ == '__main__':
  crawl_all_ssc()