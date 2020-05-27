#
import _config
from collections import defaultdict
import pandas as pd
import os

stepf2_fold = '/mnt/c/Users/maxws/Downloads/StepF2/Songs/'

'''
  Crawling
'''
def crawl_all_ssc():
  '''
    Find all .ssc files in child folders of stepf2_fold
    /stepf2_fold/18-PRIME 2/15A2 - Start on Red/15A2 - Start On Red - Nato.ssc
  '''
  print(f'Crawling local .sscs in {stepf2_fold} ...')
  get_sub_folds = lambda f: [os.path.join(f, s) for s in os.listdir(f) if os.path.isdir(os.path.join(f, s))]
  ssc_matcher = '.ssc'
  get_ssc = lambda f: [os.path.join(f, s) for s in os.listdir(f) if s[-len(ssc_matcher):] == ssc_matcher]

  sfolds = get_sub_folds(stepf2_fold)
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

'''
  StepF2 DataFrame
'''
stepf2_df_fn = _config.DATA_DIR + f'local_stepf2_files.csv'
if not os.path.isfile(stepf2_df_fn):
  crawl_all_ssc()
stepf2_df = pd.read_csv(stepf2_df_fn, index_col = 0)

standard_packs = [
  '1ST~3RD',
  'BasicModeGroup',
  'EXCEED~ZERO',
  'FIESTA 2',
  'FIESTA EX',
  'FIESTA',
  'INFINITY',
  'JUMP',
  'NX ABSOLUTE',
  'NX2',
  'PRIME 2',
  'PRIME',
  'PRIME2 [BETA]',
  'PRO~PRO2',
  'REBIRTH~PREX 3',
  'S.E.~EXTRA',
  # 'SKILLUP ZONE',
  'XX',
]

modern_packs = [
  'FIESTA 2',
  'PRIME',
  'PRIME 2',
  'PRIME2 [BETA]', 
  'XX',
]

datasets = {
  'all': stepf2_df,
  'standard': stepf2_df[stepf2_df['Pack'].isin(standard_packs)],
  'postf': stepf2_df[stepf2_df['Pack'].isin(modern_packs)],
}

'''
  .ssc class
'''
class SSCFile():
  '''
    .ssc file format
    https://github.com/stepmania/stepmania/wiki/ssc
    https://github.com/stepmania/stepmania/wiki/sm
  '''
  def __init__(self, ssc_fn):
    self.ssc_fn = ssc_fn
    self.attributes = dict()
    self.stepcharts = []

    lines = open(ssc_fn, 'r').readlines()
    file_sections = self.__parse_file_sections(lines)

    self.attributes = self.__parse_attributes(file_sections['global_header'])
    self.__parse_stepcharts(file_sections['stepcharts'])

    # TODO: Make basic stats dict
    pass

  '''
    Parsing
  '''
  def __parse_file_sections(self, lines):
    '''
      Delineated by #NOTEDATA:;
      Returns fs = {
        'global_header': str with \n,
        'stepcharts': list of strs with \n,
      }
    '''
    delim = '#NOTEDATA:;'
    all_lines = '\n'.join(lines)
    sections = all_lines.split(delim)

    assert '#TITLE' in sections[0], f'Error in parsing {self.ssc_fn}'
    for section in sections[1:]:
      assert '#STEPSTYPE' in section, f'Error in parsing {self.ssc_fn}'

    return {
      'global_header': sections[0],
      'stepcharts': sections[1:],
    }

  def __parse_stepcharts(self, stepcharts):
    delim = '#NOTES:\n'
    for stepchart in stepcharts:
      [header, notes] = stepchart.split(delim)[:2]
      attributes = self.__parse_attributes(header)
      stepchart_nm = self.__stepchart_name(attributes)
      notes = self.__parse_notes()
    # TODO: Aggregate attribute and notes across stepcharts and return something
    return

  def __parse_attributes(self, header):
    '''
      Input header is a string with \n
  
      Designed format
      <key>:<value>, potentially \n, ending with ;
    '''
    kvs = header.replace('\n', '').split(';')
    remove_comments = lambda s: s.split('//')[0]

    atts = dict()
    for kv in kvs:
      kv = remove_comments(kv)
      if ':' in kv:
        [key, val] = kv.split(':')[:2]
        atts[key] = val
    return atts

  def __parse_notes(self):
    '''
      TODO
    '''

    return

  '''
   Support
  '''
  def __stepchart_name(self, atts):
    '''
    '''
    getter = lambda q: atts[q] if q in atts else ''
    stepstype = getter('STEPSTYPE')
    level = getter('METER')
    desc = getter('DESCRIPTION')
    chartname = getter('CHARTNAME')
    name = f'{stepstype} {level} {desc} {chartname}'
    return name

