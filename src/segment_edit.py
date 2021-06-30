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

def manual_override_segment(sc_nm, beats, features, annots, motifs):

  if scinfo.name_to_level[sc_nm] < _params.min_footswitch_level:
    return force_jacks(beats, features, annots, motifs)

  if sc_nm in overrides:
    print(f'Found manual override for {sc_nm}!')
    return overrides[sc_nm](beats, features, annots, motifs)
  return annots, motifs


'''
  Force specific movement
'''
def force_jacks(beats, features, annots, motifs):
  def jackfootswitch(annots, beat, ft):
    if ft[feature_mapper['repeated line']]:
      annots[beat] = 'jack'
  for beat, ft in zip(beats, features):
    jackfootswitch(annots, beat, ft)
  return annots, motifs


def force_footswitch(beats, features, annots, motifs):
  def jackfootswitch(annots, beat, ft):
    if ft[feature_mapper['repeated line']]:
      annots[beat] = 'footswitch'

  for beat, ft in zip(beats, features):
    jackfootswitch(annots, beat, ft)
  return annots, motifs


'''
  Chart-specific overrides
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


overrides = {
  'Native - SHK S20 arcade': native_s20,
  'Loki - Lotze S21 arcade': force_footswitch,
}