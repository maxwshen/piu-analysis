'''
  Tagging - deprecated?
'''
def tag_chart(context_df, nm):
  '''
    Tags and adjectives are sorted by score
  '''
  row = context_df[context_df['Name (unique)'] == nm].iloc[0]
  found_tags = {}
  for tag in get_tags(context_df):
    keep, ranker = get_stats(row, context_df, tag)
    if keep:
      found_tags[tag] = ranker

  tags = sorted(found_tags, key=found_tags.get, reverse=True)
  tag_to_adjs = dict()
  for tag in tags:
    tag_to_adjs[tag] = get_adjectives(row, context_df, tag)

  return tag_to_adjs


def get_tags(df):
  '''
    Identify base tags -- everything with kw except exclude list
  '''
  kw = ' - 50% nps'
  tags = [col.replace(kw, '') for col in df.columns if kw in col]
  exclude = {
    'Twist angle - none',
    'Twist solo diagonal',
    'Twist angle - 90',
    'Twist angle - close diagonal',
    'Twist angle - far diagonal',
    'Twist angle - 180',
    'Irregular rhythm',
    'Rhythm change',
  }
  return [tag for tag in tags if tag not in exclude]


def get_stats(row, context_df, tag, verbose = False):
  '''
    Include tag if (OR)
    - Frequency is in top 80% percentile of context charts
    - Frequency is above 10%
  '''
  PCT_THRESHOLD = 0.80
  OBJECTIVE_MIN_FQ = 0.10
  suffix_to_adjective = {
    ' - frequency': '',
  }
  col = f'{tag} - frequency'
  val, context = row[col], context_df[col]
  pct = sum(context < val) / len(context)

  keep = False
  ranker = 0
  if pct >= PCT_THRESHOLD or val >= OBJECTIVE_MIN_FQ:
    keep = True
    ranker = pct

  if verbose:
    print(col.ljust(30), f'{val:.2f} {pct:.0%}')
  return keep, ranker


def get_adjectives(row, context_df, tag, verbose = False):
  '''
    Include these adjectives only for tags included for other reasons
    Adjectives are sorted by score
  '''
  adjs = dict()
  adjs.update(twistiness(row, context_df, tag, verbose))
  adjs.update(travel(row, context_df, tag, verbose))
  adjs.update(speed(row, context_df, tag, verbose))
  # adjs.update(length(row, context_df, tag, verbose))
  adjs.update(rhythm(row, context_df, tag, verbose))

  adj_kws = sorted(adjs, key=adjs.get, reverse=True)
  return adj_kws


def speed(row, context_df, tag, verbose):
  '''
    Label 'fast' only if (AND)
    - speed (80% nps or nps of longest) is above average within chart
    - speed is high percentile among context charts
    - speed is above a minimum interesting nps
    - frequency of movement pattern is in top 50th percentile
  '''
  speed_cols = [
    # ' - 80% nps',
    ' - median nps of longest'
  ]

  fq_col = f'{tag} - frequency'
  fq = row[fq_col]
  # fq_pct = sum(context_df[fq_col] < fq) / len(context_df)

  adjs = dict()
  for suffix in speed_cols:
    col = f'{tag}{suffix}'
    if col in row.index:
      val, context = row[col], context_df[col]
      pct = sum(context < val) / len(context)
      nps_str = f'{val:.1f} nps'
      adjs[nps_str] = pct

      ebpm_str = effective_bpm(val, row['BPM mode'])
      adjs[ebpm_str] = pct
      # print(ebpm_str)
      # import code; code.interact(local=dict(globals(), **locals()))

      if verbose:
        print(col.ljust(30), f'{val:.2f} {pct:.0%}')
  return adjs


def effective_bpm(nps, base_bpm):
  npm = nps * 60

  # Get lower and upper bpm, ensure that 2x does not fit in
  lower = base_bpm * 2/3
  upper = base_bpm * 4/3
  while lower * 2 < upper:
    lower += 1
  
  factors = {
    4:    'whole note',
    2:    'half note',
    1:    'quarter note',
    1/2:  '8th note',
    # 1/3:  '12th note',
    1/4:  '16th note',
    # 1/6:  '24th note',
    1/8:  '32nd note',
    # 1/12: '48th note',
    1/16: '64th note',
  }
  for factor in factors:
    ebpm = npm * factor
    if lower <= ebpm <= upper:
      break
  note_type = factors[factor]
  ebpm_str = f'{note_type}s at {ebpm:.0f} bpm'
  return ebpm_str


def length(row, context_df, tag, verbose):
  # Add 'long' using max len sec and nps of longest
  PCT_THRESHOLD = 0.80
  MIN_INTERESTING_NPS = 5   # todo - level dependent
  mean_nps = row['Notes per second since downpress - mean']
  adjs = dict()
  col = f'{tag} - max len sec'
  if col in row.index:
    val, context = row[col], context_df[col]
    pct = sum(context < val) / len(context)
    nps = row[f'{tag} - mean nps of longest']
    if pct >= PCT_THRESHOLD and nps >= MIN_INTERESTING_NPS and nps >= mean_nps:
      adjs['long'] = pct
  return adjs


def twistiness(row, context_df, tag, verbose):
  PCT_THRESHOLD = 0.65
  suffix_to_adjective = {
    ' - % 180 twist': '180 twists',
    ' - % far diagonal+ twist': 'hard diagonal twists',
    ' - % diagonal+ twist': 'diagonal twists',
    ' - % 90+ twist': 'twists',
    ' - % no twist': 'front-facing',
  }

  adjs = dict()
  for suffix, adjective in suffix_to_adjective.items():
    col = f'{tag}{suffix}'
    if col in row.index:
      val, context = row[col], context_df[col]
      pct = sum(context < val) / len(context)
      if verbose:
        print(col.ljust(30), f'{val:.2f} {pct:.0%}', )
      if pct >= PCT_THRESHOLD:
        adjs[adjective] = pct
        # Only add top twistiness adjective
        break
  return adjs


def travel(row, context_df, tag, verbose):
  PCT_THRESHOLD = 0.75
  high_adjective = 'large movements'
  low_adjective = 'small movements'
  suffixes = [
    ' - mean travel (mm)',
    ' - 80% travel (mm)',
  ]
  adjs = dict()
  for suffix in suffixes:
    col = f'{tag}{suffix}'
    if col in row.index:
      val, context = row[col], context_df[col]
      pct = sum(context < val) / len(context)
      if verbose:
        print(col.ljust(30), f'{val:.2f} {pct:.0%}', )
      if pct >= PCT_THRESHOLD:
        adjs[high_adjective] = pct
      if pct <= 1 - PCT_THRESHOLD:
        adjs[low_adjective] = 1 - pct
  return adjs


def rhythm(row, context_df, tag, verbose):
  PCT_THRESHOLD = 0.90
  high_adjective = 'irregular rhythm'
  low_adjective = 'consistent rhythm'
  suffixes = [
    # ' - % irregular rhythm',
    ' - % rhythm change',
  ]
  adjs = dict()
  for suffix in suffixes:
    col = f'{tag}{suffix}'
    if col in row.index:
      val, context = row[col], context_df[col]
      pct = sum(context < val) / len(context)
      if verbose:
        print(col.ljust(30), f'{val:.2f} {pct:.0%}', )
      if pct >= PCT_THRESHOLD:
        adjs[high_adjective] = pct
      # if pct <= 1 - PCT_THRESHOLD:
        # adjs[low_adjective] = 1 - pct
  return adjs

