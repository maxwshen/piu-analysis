'''
    Featurize line nodes for segmentation
'''

from numpy.lib.function_base import disp
import _config, _data, _stances, util, _params
import sys, os, pickle, fnmatch, datetime, subprocess, functools, copy
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Set, Tuple
from heapq import heappush, heappop

import _movement, _params, _memoizer, _stances, _notelines, _stepcharts
import _graph, b_graph, segment_edit, _qsub

# Default params
inp_dir = _config.OUT_PLACE + f'b_graph/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'
util.ensure_dir_exists(out_dir)

scinfo = _stepcharts.SCInfo()

feature_mapper = {
  'hit 1': 0,
  'hit 2': 1,
  'hit 3+': 2,
  'active hold': 3,
  'repeated line': 4,
  'beat since': 5,
}
def featurize(line_nodes, downpress_nodes):
  '''
    All input nodes are not multilines and have a downpress
    Returns a list of features, indexed by downpress nodes
  '''
  print('Featurizing ...')
  xs = []
  prev_line = ''
  prev_beat = np.nan
  beats = []
  for dpn in downpress_nodes:
    x = np.zeros(len(feature_mapper))
    d = line_nodes[dpn]
    line = d['Line with active holds']
    beat = d['Beat']

    if _notelines.num_downpress(line) == 1:
      x[feature_mapper['hit 1']] = 1
    elif _notelines.num_downpress(line) == 2:
      x[feature_mapper['hit 2']] = 1
    elif _notelines.num_downpress(line) >= 3:
      x[feature_mapper['hit 3+']] = 1

    if '3' in line or '4' in line:
      x[feature_mapper['active hold']] = 1

    if prev_line == line:
      x[feature_mapper['repeated line']] = 1
    
    x[feature_mapper['beat since']] = beat - prev_beat

    prev_line = line
    prev_beat = beat
    xs.append(x)
    beats.append(d['Beat'])

  assert beats == sorted(beats), 'ERROR: Beats is not sorted by default'
  return xs, beats


def uniform_segment(features, beats):
  '''
    Finds beat sections with uniform rhythm and features
    Two passes:
    1. Find 3+ consecutive lines with uniform features and rhythm
      Collect beat increments
      Optional TODO: Filter beat increments by level?
    2. Find 2 consecutive lines with uniform features and rhythms matching above
  '''

  def has_uniform_features(i, j):
    # compare "beat since" at i+1, not i
    if features[i+1][feature_mapper['repeated line']]:
      features_match = all(features[i+1][:-1] == features[j][:-1])
      prec_match = all(features[i+1][-2:] == features[j][-2:])
      init_ok = features[i][feature_mapper['hit 1']]
      result = features_match and prec_match and init_ok
    else:
      features_match = all(features[i][:-2] == features[j][:-2])
      prec_match = all(features[i+1][-2:] == features[j][-2:])
      result = features_match and prec_match
    return result

  def get_uniform_segments(min_lines, beat_increments):
    obs_incs = set()
    sections = []
    i, j = 0, 1
    while i < len(beats) and j < len(beats):
      if not has_uniform_features(i, j):
        if j - i >= min_lines:
          beat_inc = features[i+1][-1]
          if not beat_increments:
            sections.append((beats[i], beats[j-1]))
            obs_incs.add(beat_inc)
            i = j - 1
            j = i + 1
          else:
            # if beat_inc in beat_increments:
            if beat_inc <= max(beat_increments):
              sections.append((beats[i], beats[j-1]))
              obs_incs.add(beat_inc)
              i = j - 1
              j = i + 1
            else:
              i = i + 1
              j = i + 1              
        else:
          i = i + 1
          j = i + 1
      else:
        j += 1
    if j - i >= min_lines:
      sections.append((beats[i], beats[j-1]))
    return sections, obs_incs

  print('Finding uniform segments ...')
  sections, obs_incs = get_uniform_segments(3, None)
  two_sections, _ = get_uniform_segments(2, obs_incs)
  return sorted(list(set(sections + two_sections)))


'''
  Motifs
'''
def find_motifs(line_nodes, features, beats):
  '''
    Find enriched rhythm/feature motifs in lines outside of uniform sections
    First, make motifs for active holds
    Then, find motifs in remaining sections
    Returns a dict. motif_id: list of beat sections.
  '''
  print('Finding motifs ...')
  sections = []
  MIN_MOTIF_LEN = 4
  INIT_SEED_LEN = 16
  lines = [line_nodes[k]['Line with active holds'] for k in line_nodes
            if line_nodes[k]['Beat'] in beats]
  all_motifs = get_active_hold_motifs(lines, features, beats)

  for k, v in all_motifs.items():
    for section in v:
      sections.append(section)

  motif_len = INIT_SEED_LEN
  while motif_len >= MIN_MOTIF_LEN:
    motifs = get_enriched_motifs(lines, features, beats, motif_len, sections)

    all_motifs.update(motifs)
    for k, v in motifs.items():
      for section in v:
        sections.append(section)
    motif_len -= 1

  return all_motifs


def get_active_hold_motifs(lines, features, beats):
  # Annotates contiguous holds: 2 -> 4 -> 3
  # Starts on first 1 or 2
  all_motifs = {}
  i, j = 0, 1
  in_hold = False
  found_hit = False
  num_found = 0
  # import code; code.interact(local=dict(globals(), **locals()))
  while i < len(beats) and j < len(beats):
    # print(i, j)
    if in_hold:
      if '4' in lines[j]:
        j += 1
      elif '3' in lines[j] and '4' not in lines[j]:
        all_motifs[f'hold-{num_found}'] = [(beats[i], beats[j-1])]
        in_hold = False
        i, j = j+1, j+2
        num_found += 1
      else:
        print('\n Warning: Detected unresolved hold')
        print(lines[i:j+1])
        print(beats[i:j+1])
        # import code; code.interact(local=dict(globals(), **locals()))
        # raise Exception('Detected unresolved hold')
        j += 1
    else:
      if any(x in lines[i] for x in list('24')) and '1' in lines[i]:
        in_hold = True
      elif any(x in lines[i] for x in list('4')) and '2' in lines[i]:
        in_hold = True
      elif '3' in lines[i] and any(x in lines[i] for x in '12'):
        all_motifs[f'hold-{num_found}'] = [(beats[i], beats[i])]
        num_found += 1
        i += 1
        j += 1
      else:
        i += 1
        j += 1

  return all_motifs


def get_enriched_motifs(lines, features, beats, motif_len, sections):
  '''
    Returns a dict. motif_id: list of beat sections.
    Ignores sections
    Reject motifs that start or end in active holds
  '''
  from collections import defaultdict
  dd = defaultdict(list)
  for i in range(len(beats) - motif_len):
    j = i + motif_len
    if range_in_sections(beats[i], beats[j], sections):
      continue
    
    if any(x in lines[i] for x in list('34')):
      continue

    motif = str(np.concatenate([features[i][:4]] + features[i+1:j]))
    if not beat_in_any_section(beats[i], dd[motif]):
      if not beat_in_any_section(beats[j], dd[motif]):
        dd[motif].append((beats[i], beats[j-1]))

  # Filter to enriched motifs
  if motif_len >= 8:
    threshold = 2
  else:
    threshold = 3
  filt_dd = {k: v for k, v in dd.items() if len(v) >= threshold}
  renamed_dd = rename_keys(motif_len, filt_dd)

  # Extend seeds and prune
  extended_dd = extend_seeds_and_prune(lines, features, beats,
      renamed_dd, motif_len)
  return extended_dd


def extend_seeds_and_prune(lines, features, beats, motifs, motif_len):
  '''
    When seeds are shorter than the true motif length, we get many overlapping seeds. Extend the first seed, and remove redundant overlapping seeds.
    Note: In the output, motif instances can overlap.
    Reject motifs that end in active holds
  '''
  extended_motifs = {}
  for key in motifs:
    add_len = 0
    if not redundant_seed(motifs[key], extended_motifs):
      last_beat_idxs = [beats.index(r[-1]) for r in motifs[key]]
      next_beat_idxs = [b + 1 for b in last_beat_idxs]
      if len(beats) in next_beat_idxs:
        continue
      next_features = set([tuple(features[i]) for i in next_beat_idxs])
      while len(next_features) == 1:
        last_beat_idxs = next_beat_idxs
        next_beat_idxs = [b + 1 for b in last_beat_idxs]
        if len(beats) in next_beat_idxs:
          break
        next_features = set([tuple(features[i]) for i in next_beat_idxs])
        add_len += 1

      # Reject motifs that end in active hold
      last_line = lines[last_beat_idxs[0]]
      if any(x in last_line for x in list('24')):
        continue

      extended_rs = [(r[0], beats[i]) for r, i in zip(motifs[key],
                                                      last_beat_idxs)]
      rand_id = get_rand_string(8)
      extended_motifs[f'{motif_len+add_len}-{rand_id}'] = extended_rs

  return extended_motifs


'''
  Motifs - helper functions
'''
def rename_keys(name, dd):
  renamed_dd = {}
  for i, (k, v) in enumerate(dd.items()):
    renamed_dd[f'{name}-{i}'] = v
  return renamed_dd


def has_overlap(motifs):
  ranges = []
  for motif in motifs:
    for r in motifs[motif]:
      if range_in_sections(r[0], r[1], ranges):
        return True
      else:
        ranges.append(r)
  return False


def get_rand_string(length):
  import string, random
  letters = string.ascii_letters
  return ''.join(random.choice(letters) for i in range(length))


def redundant_seed(seed_range, extended_motifs):
  return any([all_ranges_in_ranges(seed_range, v) for k, v in extended_motifs.items()])


def all_ranges_in_ranges(rs1, rs2):
  return all([range_in_range(r1, r2) for r1, r2 in zip(rs1, rs2)])


def range_in_range(r1, r2):
  return r2[0] <= r1[0] <= r1[1] <= r2[1]


def increment_beat(beat, beats):
  idx = beats.index(beat)
  if idx != len(beats) - 1:
    return beats[idx + 1]
  else:
    return None


'''
  Data structure
'''
def form_data_struct(sc_nm, line_nodes, features, beats, uniform_sections, all_motifs):
  '''
    d[beat] = decision in {
      'alternate',
      'jacks',
      'footswitch',
      'bracket',
      'jump',
      'jumporbracket',
      missing: (let dijkstra decide) 
    }
  '''
  level = scinfo.name_to_level[sc_nm]
  unif_ds = struct_uniform(line_nodes, features, beats, uniform_sections, level)
  motif_ds = struct_motifs(line_nodes, features, beats, all_motifs, level)
  joint_ds = unif_ds.update(motif_ds)
  return unif_ds, motif_ds


def struct_uniform(line_nodes, features, beats, uniform_sections, level):
  ds = {}
  for usec in uniform_sections:
    bs = get_key_in_section(line_nodes, beats, usec, 'Beat')
    idxs = [beats.index(b) for b in bs]
    fts = [features[i] for i in idxs]
    present_fts = [ft[:-2] for ft in fts]
    lines = get_key_in_section(line_nodes, beats, usec, 'Line')
    lines_holds = get_key_in_section(line_nodes, beats, usec, 'Line with active holds')

    if single_hit_alternating(lines_holds, fts):
      annot = 'alternate'
    # Decide 2hits as jumps or brackets
    elif is_twohit(lines_holds, fts):
      if level < _params.bracket_level_threshold:
        annot = 'jump'
      elif _notelines.frac_bracketable(lines) < 1:
        annot = 'jump'
      else:
        annot = 'jumporbracket'
    elif uniform_jacks(lines, fts):
      annot = 'jackorfootswitch'
    else:
      annot = None

    if annot:
      for b in bs[1:]:
        ds[b] = annot

  # Annotate single twohits
  for beat, ft in zip(beats, features):
    lines_holds = get_key_in_section(line_nodes, beats, (beat, beat), 'Line with active holds')
    lines = get_key_in_section(line_nodes, beats, (beat, beat), 'Line')
    if is_twohit(lines_holds, [ft]) and beat not in ds:
      if level < _params.bracket_level_threshold:
        ds[beat] = 'jump'
      elif _notelines.frac_bracketable(lines) == 0:
        ds[beat] = 'jump'
      else:
        ds[beat] = 'jumporbracket'
    
  # Alternate 3->1/2 when not same pad
  for b1, b2 in zip(beats[:-1], beats[1:]):
    lines_holds = get_key_in_section(line_nodes, beats, (b1, b2), 'Line with active holds')
    [line1, line2] = lines_holds
    line_len = len(lines_holds[0])
    if line2.replace('2', '3').replace('1', '3') != line1:
      if line1.count('0') == line_len-1 and '3' in line1:
        if line2.count('0') == line_len-1 and any(x in line2 for x in list('12')):
          ds[b2] = 'alternate'

  # Force same foot on 1->2 with same pad
  for b1, b2 in zip(beats[:-1], beats[1:]):
    lines_holds = get_key_in_section(line_nodes, beats, (b1, b2), 'Line with active holds')
    [line1, line2] = lines_holds
    line_len = len(lines_holds[0])
    if line2.replace('2', '1') == line1:
      if line1.count('0') == line_len-1 and '1' in line1:
        if line2.count('0') == line_len-1 and '2' in line2:
          ds[b2] = 'same'

  return ds


def struct_motifs(line_nodes, features, beats, motifs, level):
  '''
    Local consistency within motifs. Do not compare across motif instances.
  '''
  ds = {}
  for motif in motifs:
    for section in motifs[motif]:
      lines = get_key_in_section(line_nodes, beats, section, 'Line')
      lines_holds = get_key_in_section(line_nodes, beats, section, 'Line with active holds')
      bs = get_key_in_section(line_nodes, beats, section, 'Beat')
      fts = [features[beats.index(b)] for b in bs]

      jfs, twohit, hold = 'any', 'any', 'any'
      # 2-hits
      twohitlines = get_twohit_lines(lines_holds, fts)
      if len(twohitlines) > 0:
        if level < _params.bracket_level_threshold:
          twohit = 'jump'
        elif _notelines.frac_bracketable(twohitlines) < 1:
          twohit = 'jump'
        else:
          twohit = 'jumporbracket'

      # Jacks
      if num_jacks(lines, fts) > 0:
        jfs = 'jackorfootswitch'

      # Hold
      lines_holds = get_key_in_section(line_nodes, beats, section,
          'Line with active holds')
      if any(bool('4' in lineh or '3' in lineh) for lineh in lines_holds):
        if level < _params.hold_bracket_level_threshold:
          hold = 'jack'
        else:
          has_dp = lambda line: any(x in line for x in list('12'))
          dp_lines = [line for line in lines_holds if has_dp(line)]
          dp_count = lambda line: line.count('1') + line.count('2')
          num_dps = sum(dp_count(line) for line in dp_lines)
          if len(dp_lines) == 1 and num_dps == 1:
            hold = 'jack'
          else:
            hold = 'jackoralternateorfree'
      
      ds[section] = f'{jfs}-{twohit}-{hold}'
  return ds



'''
  Struct uniform - helper functions
'''
def single_hit_alternating(lines_with_holds, fts):
  # Truncate beat since
  if len(lines_with_holds) == 2:
    # Do not alternate on 1->2 on same pad, and 3->2 on same pad
    [line1, line2] = lines_with_holds
    if line1 == line2.replace('2', '1'):
      return False
    if line1 == line2.replace('2', '3'):
      return False
  trunc_fts = [ft[:-1] for ft in fts[1:]]
  signature = np.array([1, 0, 0, 0, 0])
  matches_sig = lambda x: all(x == signature)
  return all(matches_sig(tft) for tft in trunc_fts)


def is_twohit(lines_with_holds, fts):
  # Must not be in a hold
  twohit = fts[0][feature_mapper['hit 2']]
  hold = any([any([x in line for x in list('34')]) for line in lines_with_holds])
  return twohit and not hold


def uniform_jacks(lines, fts):
  '''
    Must be downpress of 1 and have repeated lines.
    Assumes lines and fts are from a uniform section.
  '''
  repeats = fts[1][feature_mapper['repeated line']]
  single_hit = fts[0][feature_mapper['hit 1']]
  downpress1 = '1' in lines[0]
  return repeats and single_hit and downpress1


def num_jacks(lines, fts):
  '''
    Num. jacks in lines/fts. Does not assume uniform section.
  '''
  isjack = lambda line, ft: ft[feature_mapper['repeated line']] and \
                            ft[feature_mapper['hit 1']] and \
                            '1' in line
  return sum(isjack(line, ft) for line, ft in zip(lines, fts))


def get_twohit_lines(lines_holds, fts):
  return [line for line, ft in zip(lines_holds, fts) if is_twohit([line], [ft])]


'''
  Helper
'''
def get_key_in_section(line_nodes, beats, section, key):
  return [line_nodes[k][key] for k in line_nodes
          if beat_in_section(line_nodes[k]['Beat'], section)]


def beat_in_section(beat, section):
  return section[0] <= beat <= section[1]


def beat_in_any_section(beat, sections):
  return any([beat_in_section(beat, sec) for sec in sections])


def range_in_sections(beat1, beat2, sections):
  for start, end in sections:
    if beat1 <= start <= beat2 or beat1 <= end <= beat2:
      return True
    if start <= beat1 <= end or start <= beat2 <= end:
      return True
  return False


def filter_lines_by_section(features, beats, uniform_sections):
  filt_features, filt_beats = [], []
  for feature, beat in zip(features, beats):
    if not beat_in_any_section(beat, uniform_sections):
      filt_features.append(feature)
      filt_beats.append(beat)
  return filt_features, filt_beats


def calc_coverage(beats, uniform_sections, all_motifs):
  '''
    Calculates fraction of notes that are covered
  '''
  covered_beats = set()
  for beat in beats:
    if beat_in_any_section(beat, uniform_sections):
      covered_beats.add(beat)
    else:
      for motif in all_motifs:
        if beat_in_any_section(beat, all_motifs[motif]):
          covered_beats.add(beat)
          break
        
  covered = len(covered_beats)
  print(f'{covered / len(beats):.1%}, {covered}/{len(beats)} lines covered')

  uncovered = [beat for beat in beats if beat not in covered_beats]
  return uncovered


def calc_coverage_ds(beats, ds):
  covered_beats = set(b for b in beats if b in ds)
  covered = len(covered_beats)
  uncovered = [b for b in beats if b not in ds]
  print(f'{covered / len(beats):.1%}, {covered}/{len(beats)} lines covered')
  return uncovered


'''
  IO
'''
def save_annotations(sc_nm, line_nodes, unif_ds, motif_ds):
  dd = defaultdict(list)
  for line in line_nodes:
    if line not in ['init', 'final']:
      for k, v in line_nodes[line].items():
        dd[k].append(v)
    
      beat = line_nodes[line]['Beat']
      annot = unif_ds[beat] if beat in unif_ds else ''
      dd['Annotation'].append(annot)

      motif_annot = ''
      for sec in motif_ds:
        if beat_in_section(beat, sec):
          motif_annot = (sec, motif_ds[sec])
      dd['Motif'].append(motif_annot)

  df = pd.DataFrame(dd)

  # Check beat and time are monotonically ascending
  require_monotonic = ['Beat', 'Time']
  for col in require_monotonic:
    data = list(df[col])
    if data != sorted(data):
      print(f'ERROR: {col} not monotonic')
      # raise Exception()

  df['Line'] = [_notelines.excel_refmt(s) for s in df['Line']]
  df['Line with active holds'] = [_notelines.excel_refmt(s) for s in df['Line with active holds']]
  df.to_csv(out_dir + f'{sc_nm}.csv')

  with open(out_dir + f'{sc_nm}-uniform.pkl', 'wb') as f:
    pickle.dump(unif_ds, f)

  with open(out_dir + f'{sc_nm}-motif.pkl', 'wb') as f:
    pickle.dump(motif_ds, f)
  return


def load_annotations(fold, sc_nm):
  fn = fold + f'{sc_nm}-uniform.pkl'
  if not os.path.isfile(fn):
    print(f'ERROR: File not found in segment')
    raise Exception(f'ERROR: File not found in segment')

  with open(fold + f'{sc_nm}-uniform.pkl', 'rb') as f:
    unif_d = pickle.load(f)
  with open(fold + f'{sc_nm}-motif.pkl', 'rb') as f:
    motif_d = pickle.load(f)
  return unif_d, motif_d


def filter_annots(beats, unif_d, motifs):
  '''
    Remove any-any-any motifs
    Remove motifs without any compatible annotations.
    Called in _graph init
  '''
  new_motifs = {}
  for section in motifs:
    tag = motifs[section]
    if tag != 'any-any-any':
      [tag_jfs, tag_twohits, tag_hold] = tag.split('-') 
      bs = [b for b in beats if section[0] <= b <= section[1]]
      annots = [unif_d[b] for b in bs if b in unif_d]
      if tag_jfs in ['jack', 'footswitch', 'jackorfootswitch']:
        if 'jackorfootswitch' in annots:
          new_motifs[section] = tag
      if tag_twohits in ['jump', 'bracket', 'jumporbracket']:
        if 'jumporbracket' in annots:
          new_motifs[section] = tag
      if tag_hold in ['jack', 'alternate', 'free', 'jackoralternateorfree']:
        new_motifs[section] = tag
  return unif_d, new_motifs


'''
  Run
'''
def run_single(nm):
  # Load lines
  line_nodes, line_edges_out, line_edges_in = b_graph.load_data(inp_dir, nm)

  # Remove multihit nodes
  ks = list(line_nodes.keys())
  for node in ks:
    if any(x in node for x in ['multi', 'init', 'final']):
      del line_nodes[node]

  downpress_filter = lambda node: 'multi' not in node and \
                             'init'  not in node and \
                             'final' not in node and \
                             _notelines.has_downpress(line_nodes[node]['Line'])
  downpress_nodes = [node for node in line_nodes if downpress_filter(node)]

  print(f'{nm}: {len(downpress_nodes)} nodes')

  dp_features, dp_beats = featurize(line_nodes, downpress_nodes)
  uniform_sections = uniform_segment(dp_features, dp_beats)
  print('Coverage with uniform sections:')
  calc_coverage(dp_beats, uniform_sections, {})

  filter = lambda node: 'multi' not in node and \
                        'init'  not in node and \
                        'final' not in node
  all_nodes = [node for node in line_nodes if filter(node)]
  features, beats = featurize(line_nodes, all_nodes)

  # Form and save data structure
  all_motifs = find_motifs(line_nodes, features, beats)
  unif_ds, motif_ds = form_data_struct(nm, line_nodes,
      features, beats, uniform_sections, all_motifs)
  # print('Coverage with uniform sections + motifs:')
  # uncovered = calc_coverage_ds(beats, unif_ds)

  unif_ds, motif_ds = segment_edit.manual_override_segment(nm,
      beats, features, unif_ds, motif_ds)
  save_annotations(nm, line_nodes, unif_ds, motif_ds)
  return


@util.time_dec
def main():
  print(NAME)
  
  # Test: Single stepchart
  # nm = 'Super Fantasy - SHK S16 arcade'
  # nm = 'Super Fantasy - SHK S19 arcade'
  # nm = 'London Bridge - SCI Guyz S11 arcade'
  # nm = 'Phalanx "RS2018 Edit" - Cranky S22 arcade'
  # nm = 'Xeroize - FE S24 arcade'
  # nm = 'Tepris - Doin S17 arcade'
  # nm = 'Last Rebirth - SHK S15 arcade'
  # nm = 'The End of the World ft. Skizzo - MonstDeath S20 arcade'
  # nm = '8 6 - DASU S20 arcade'
  # nm = 'Loki - Lotze S21 arcade'
  # nm = 'King of Sales - Norazo S21 arcade'
  # nm = 'Follow me - SHK S9 arcade'
  # nm = 'Death Moon - SHK S22 shortcut'
  # nm = 'Fresh - Aspektz S14 arcade infinity'
  # nm = 'Phalanx "RS2018 Edit" - Cranky S22 arcade'
  # nm = 'Chicken Wing - BanYa S7 arcade'
  # nm = 'CROSS SOUL - HyuN feat. Syepias S8 arcade'
  # nm = 'Native - SHK S20 arcade'
  # nm = 'Sorceress Elise - YAHPP S23 arcade'
  # nm = 'Chicken Wing - BanYa S17 arcade'
  # nm = 'Hypnosis - BanYa S18 arcade'
  # nm = 'NoNoNo - Apink S14 arcade'
  # nm = 'Rage of Fire - MAX S16 arcade'
  # nm = 'Conflict - Siromaru + Cranky S17 arcade'
  # nm = 'Final Audition - BanYa S15 arcade'
  # nm = 'Oy Oy Oy - BanYa S13 arcade'
  # nm = 'An Interesting View - BanYa S13 arcade'
  # nm = 'Bee - BanYa S15 arcade'
  # nm = 'Beat of The War 2 - BanYa S21 arcade'
  # nm = 'Exceed2 Opening - Banya S15 shortcut'
  # nm = 'The Little Prince (Prod. Godic) - HAON, PULLIK S9 arcade'
  # nm = 'The Little Prince (Prod. Godic) - HAON, PULLIK S13 arcade'
  # nm = 'Obelisque - ESTi x M2U S17 arcade'
  # nm = 'I Want U - MAX S19 arcade'
  # nm = 'Forgotten Vampire - WyvernP S18 arcade'
  # nm = 'The Little Prince (Prod. Godic) - HAON, PULLIK S18 arcade'
  # nm = 'Prime Time - Cashew S23 remix'
  # nm = 'HYPERCUBE - MAX S15 arcade'
  # nm = 'Setsuna Trip - Last Note. S16 arcade'
  # nm = 'Uranium - Memme S19 arcade'
  # nm = 'Gothique Resonance - P4Koo S20 arcade'
  # nm = 'Club Night - Matduke S18 arcade'
  # nm = 'BANG BANG BANG - BIGBANG S15 arcade'
  # nm = 'PRIME - Tatsh S11 arcade'
  # nm = 'Shub Niggurath - Nato S24 arcade'
  # nm = 'CARMEN BUS - StaticSphere & FUGU SUISAN S12 arcade'
  # nm = 'Mr. Larpus - BanYa S22 arcade'
  # nm = 'Bad End Night - HitoshizukuP x yama S17 arcade'
  # nm = 'Super Fantasy - SHK S16 arcade'
  # nm = 'Bad Apple!! feat. Nomico - Masayoshi Minoshima S17 arcade'
  # nm = 'F(R)IEND - D_AAN S23 arcade'
  
  # Test: Many hands
  # nm = 'London Bridge - SCI Guyz S11 arcade'

  # Test: failures
  # nm = 'PRIME - Tatsh S7 arcade'
  # nm = '%X (Percent X) - Pory S17 arcade'
  # nm = 'Log In - SHK S20 arcade'
  # nm = 'Good Night - Dreamcatcher S17 arcade'
  # nm = 'Poseidon - Quree S20 arcade'
  # nm = 'Tales of Pumpnia - Applesoda S16 arcade'
  # nm = 'Death Moon - SHK S17 arcade'
  # nm = 'Elvis - AOA S15 arcade'
  # nm = 'Tales of Pumpnia - Applesoda S16 arcade'
  # nm = 'Wedding Crashers - SHORT CUT - - SHK S18 shortcut'
  # nm = 'Good Night - Dreamcatcher S17 arcade'
  # nm = 'Fly high - Dreamcatcher S15 arcade'
  # nm = 'Poseidon - Quree S20 arcade'
  # nm = 'HANN (Alone) - (G)I-DLE D17 arcade'
  # nm = 'Shub Niggurath - Nato S24 arcade'
  # nm = 'Club Night - Matduke D21 arcade'
  # nm = 'Macaron Day - HyuN D18 arcade'
  # nm = 'Scorpion King - r300k S15 arcade'
  # nm = 'Red Swan - Yahpp S18 arcade'
  # nm = 'Accident - MAX S18 arcade'

  # Test: Fake notes
  # nm = 'Club Night - Matduke S18 arcade'
  # nm = 'Good Night - Dreamcatcher S20 arcade'
  # nm = 'God Mode feat. skizzo - Nato S18 arcade'

  # Fixed
  # nm = 'Acquaintance - Outsider S17 arcade'

  # Test: Visual gimmicks
  # nm = 'Obliteration - ATAS S17 arcade'
  # nm = 'Nihilism - Another Ver. - - Nato S21 arcade'
  # nm = 'Full Moon - Dreamcatcher S22 arcade'
  # nm = 'Wedding Crashers - SHK S16 arcade'
  # nm = 'V3 - Beautiful Day S17 arcade'

  # Doubles
  # nm = 'Mitotsudaira - ETIA. D19 arcade'
  # nm = 'Canon D - BanYa D17 arcade'
  # nm = 'Shock - BEAST D15 arcade'
  # nm = 'Witch Doctor #1 - YAHPP HD19 arcade'
  # nm = 'Slam - Novasonic D19 arcade'
  # nm = 'Emperor - BanYa D17 arcade'
  # nm = 'Loki - Lotze D19 arcade'
  # nm = 'Trashy Innocence - Last Note. D16 arcade'
  # nm = '8 6 - DASU D21 arcade'
  # nm = 'Bad End Night - HitoshizukuP x yama D18 arcade'
  # nm = 'Maslo - Vospi D16 arcade'
  # nm = 'Energetic - Wanna One D19 arcade'
  # nm = 'You Got Me Crazy - MAX D18 arcade'
  # nm = 'Anguished Unmaking - void D18 arcade'
  # nm = 'Poseidon - SHORT CUT - - Quree D14 shortcut'
  nm = 'Ugly Dee - Banya Production D15 arcade'

  run_single(nm)
  return


if __name__ == '__main__':
  if len(sys.argv) == 1:
    main()
  else:
    if sys.argv[1] == 'gen_qsubs':
      _qsub.gen_qsubs(NAME, sys.argv[2])
    elif sys.argv[1] == 'run_qsubs':
      _qsub.run_qsubs(
        chart_fnm = sys.argv[2],
        start = sys.argv[3],
        end = sys.argv[4],
        run_single = run_single,
      )
    elif sys.argv[1] == 'gen_qsubs_remainder':
      _qsub.gen_qsubs_remainder(NAME, sys.argv[2], '-uniform.pkl')
    elif sys.argv[1] == 'run_single':
      run_single(sys.argv[2])