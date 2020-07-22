#
import _config
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

'''
  StepF2 DataFrame
'''
stepf2_df_fn = _config.DATA_DIR + f'local_stepf2_files.csv'
if not os.path.isfile(stepf2_df_fn):
  crawl_all_ssc()
stepf2_df = pd.read_csv(stepf2_df_fn, index_col = 0)

standard_packs = [
  '1ST~3RD',
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
  'PRO~PRO2',
  'REBIRTH~PREX 3',
  'S.E.~EXTRA',
  'XX',
  # 'BasicModeGroup',
  # 'SKILLUP ZONE',
  # 'PRIME2 [BETA]',
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
  'fiesta2_onwards': stepf2_df[stepf2_df['Pack'].isin(modern_packs)],
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
  def __init__(self, ssc_fn, pack = ''):
    self.ssc_fn = ssc_fn
    lines = open(ssc_fn, 'r').readlines()
    sections = self.__parse_file_sections(lines)

    self.global_attributes = self.__parse_attributes(sections['global_header'])
    self.global_attributes['Pack'] = pack
    self.sc_attributes, self.sc_notes = self.__parse_stepcharts(sections['stepcharts'])
    pass


  '''
    Parsing
  '''
  def __parse_file_sections(self, lines: str) -> dict:
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


  def __parse_stepcharts(self, stepcharts: list) -> list:
    delim = '#NOTES:\n'
    sc_atts = []
    all_notes = []
    for stepchart in stepcharts:
      [header, notes] = stepchart.split(delim)[:2]
      attributes = self.__parse_attributes(header)
      self.__annotate_stepchart(attributes)
      sc_atts.append(attributes)

      if '\n\n' in notes:
        notes = notes.replace('\n\n', '\n')
      notes = notes[:notes.index(';')]
      notes = notes.strip()
      all_notes.append(notes)

    return sc_atts, all_notes


  def __parse_attributes(self, header: str) -> dict:
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
        if key[0] == '#':
          key = key[1:]
        atts[key] = val
    return atts


  '''
   Support
  '''
  def __annotate_stepchart(self, atts: dict):
    '''
      Annotates custom stepchart attributes, using description and other attributes
    '''
    get_att = lambda q: atts[q] if q in atts else '-'
    song_nm = self.global_attributes['TITLE']
    stepstype = get_att('STEPSTYPE')
    bpm = get_att('BPMS')
    level = get_att('METER')
    desc = get_att('DESCRIPTION').upper()
    songtype = self.global_attributes['SONGTYPE'].lower()
    artist = self.global_attributes['ARTIST']

    if stepstype == 'pump-single':
      st_short = 'S'
    elif stepstype == 'pump-double':
      st_short = 'D'
    elif stepstype == 'pump-halfdouble':
      st_short = 'HD'
    else:
      st_short = '?'

    atts['Is UCS'] = bool('UCS' in desc)
    atts['Is quest'] = bool('QUEST' in desc)
    atts['Is hidden'] = bool('HIDDEN' in desc)
    atts['Is halfdouble'] = bool('HALF' in desc)
    if atts['Is halfdouble']:
      st_short = 'HD'
      atts['STEPSTYPE'] = 'pump-halfdouble'
    atts['Is SP'] = bool('SP' in desc)
    atts['Is DP'] = bool('DP' in desc)
    if atts['Is SP'] or atts['Is DP']:
      st_short += 'P'
    atts['Is infinity'] = bool('INFINITY' in desc)
    atts['Is train'] = bool('TRAIN' in desc)
    atts['First BPM'] = bpm.split(',')[0]

    atts['Name'] = f'{song_nm} - {artist} {st_short}{level} {songtype}'
    name_modifiers = ['Is quest', 'Is hidden', 'Is infinity', 'Is train']
    for name_mod in name_modifiers:
      if atts[name_mod]:
        nm_mod = name_mod.replace('Is ', '')
        atts['Name'] += f' {nm_mod}'

    atts['Steptype simple'] = st_short
    return


  '''
    Public
  '''
  def get_stepchart_info(self) -> pd.DataFrame:
    cols = [
      'Name',
      'Steptype simple',
      'Is UCS',
      'Is quest',
      'Is hidden',
      'Is infinity',
      'Is train',
      'Pack',
      'VERSION',
      'TITLE',
      'SUBTITLE',
      'ARTIST',
      'GENRE',
      'CREDIT',
      'OFFSET',
      'SELECTABLE',
      'SONGTYPE',
      'SONGCATEGORY',
      'TIMESIGNATURES',
      'STEPSTYPE',
      'DESCRIPTION',
      'DIFFICULTY',
      'METER',
      'DISPLAYBPM',
      'First BPM',
    ]
    '''
      Each row is a stepchart. Columns are stepchart-specific attributes and global attributes
    '''
    dd = defaultdict(list)
    for sc_att in self.sc_attributes:
      for key in cols:
        if key in sc_att:
          dd[key].append(sc_att[key])
        elif key in self.global_attributes:
          dd[key].append(self.global_attributes[key])
        else:
          dd[key].append(np.nan)
    df = pd.DataFrame(dd)
    return df

  def get_stepchart_notes(self) -> List[str]:
    '''
    '''
    return self.sc_notes
