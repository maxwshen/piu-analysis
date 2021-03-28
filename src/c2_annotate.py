
def annotate(df):
  # Annotate in forward direction
  cdd = defaultdict(list)
  for idx, row in df.iterrows():
    if idx == 0:
      cdd['Jump'].append(np.nan)
    else:
      cdd['Jump'].append(is_jump(df.iloc[idx - 1]['Stance action'], row['Stance action']))
  for col in cdd:
    df[col] = cdd[col]


'''
  Helper
'''
def is_jump(sa1, sa2):
  if sa1 == '':
    return np.nan
  stances1 = sa1[:sa1.index(';')].split(',')[:2]
  stances2 = sa2[:sa2.index(';')].split(',')[:2]
  jump_flag = True
  for s1, s2 in zip(stances1, stances2):
    if s1 == s2:
      jump_flag = False
  return 1 if jump_flag else np.nan