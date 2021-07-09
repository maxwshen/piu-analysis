'''
  Manually edit uniform section or motif annotations
  - jack or footswitch
  - jump or bracket
  manual_override_segment() called in segment.py
'''
from collections import defaultdict
import segment, _graph, _stepcharts, _params

scinfo = _stepcharts.SCInfo()

feature_mapper = {
  'hit 1': 0,
  'hit 2': 1,
  'hit 3+': 2,
  'active hold': 3,
  'repeated line': 4,
  'beat since': 5,
}

footswitch_charts = [
  'Oy Oy Oy - BanYa S13 arcade',
  'An Interesting View - BanYa S13 arcade',
  # 'Final Audition - BanYa S15 arcade',  # rerated to S18
  'Bee - BanYa S15 arcade',
  'Native - SHK S17 arcade',
  'Beat of The War 2 - BanYa S21 arcade',
  'Papasito (feat. KuTiNA) - Yakikaze & Cashew D21 arcade',
]

pure_footswitch_charts = [
  'Loki - Lotze S21 arcade',
]

def manual_override_segment(sc_nm, beats, features, annots, motifs):
  if sc_nm in overrides:
    print(f'Found manual override for {sc_nm}!')
    return overrides[sc_nm](beats, features, annots, motifs)

  if scinfo.name_to_level[sc_nm] in pure_footswitch_charts:
    print(f'Found manual override for {sc_nm}!')
    return force_repeated_line(beats, features, annots, motifs, 'footswitch')

  if scinfo.name_to_level[sc_nm] < _params.min_footswitch_level:
    print(f'Found manual override for {sc_nm}!')
    return force_repeated_line(beats, features, annots, motifs, 'jack')

  return annots, motifs


'''
  Force specific movement
'''
def force_repeated_line(beats, features, annots, motifs, force_annot):
  def jackfootswitch(annots, beat, ft):
    if ft[feature_mapper['repeated line']]:
      annots[beat] = force_annot
  for beat, ft in zip(beats, features):
    jackfootswitch(annots, beat, ft)
  return annots, motifs


'''
  Chart-specific overrides
  - Currently used for charts that mix footswitches and jacks
'''
def native_s20(beats, features, annots, motifs):
  '''
    Repeated lines with 0.75 beat since = footswitch;
      with 0.25 beat since = jack
    Jump everything
  '''
  def jackfootswitch(annots, beat, ft):
    if ft[feature_mapper['repeated line']]:
      if ft[feature_mapper['beat since']] in [0.75]:
        annots[beat] = 'footswitch'
      elif ft[feature_mapper['beat since']] in [0.25]:
        annots[beat] = 'jack'
      else:
        annots[beat] = 'jackorfootswitch'

  def jumpbracket(annots, beat, ft):
    if ft[feature_mapper['hit 2']]:
      annots[beat] = 'jump'

  for beat, ft in zip(beats, features):
    jackfootswitch(annots, beat, ft)
    jumpbracket(annots, beat, ft)
  return annots, motifs


def final_audition_s15(beats, features, annots, motifs):
  # Has jacks and footswitches; distinguish by beat since
  def jackfootswitch(annots, beat, ft):
    if ft[feature_mapper['repeated line']]:
      if ft[feature_mapper['beat since']] in [0.25]:
        annots[beat] = 'footswitch'
      elif ft[feature_mapper['beat since']] in [0.50]:
        annots[beat] = 'jack'
      else:
        annots[beat] = 'jackorfootswitch'

  for beat, ft in zip(beats, features):
    jackfootswitch(annots, beat, ft)
  return annots, motifs


def emperor_d17(beats, features, annots, motifs):
  # Hold taps should be free
  new_motifs = {}
  for k, v in motifs.items():
    [jfs, twohit, hold] = v.split('-')
    forced_hold = 'free'
    new_motifs[k] = '-'.join([jfs, twohit, forced_hold])
  
  new_annots = {}
  for beat, feature in zip(beats, features):
    if beat in annots:
      annot = annots[beat]
      if annot != 'alternate':
        new_annots[beat] = annot
  return new_annots, new_motifs


def uglydee_d15(beats, features, annots, motifs):
  # Remove all motifs
  return annots, {}


overrides = {
  'Native - SHK S20 arcade': native_s20,
  'Final Audition - BanYa S15 arcade': final_audition_s15,
  # 'Emperor - BanYa D17 arcade': emperor_d17,
  'Ugly Dee - Banya Production D15 arcade': uglydee_d15,
  'Ugly Dee - Banya Production D11 arcade': uglydee_d15,
}