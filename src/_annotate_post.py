'''
  Annotations that rely on general, local and global annotations
  - Sole diagonal twists
  - Modify drills and runs
'''

import numpy as np
import _notelines

mover = None
get_ds = None

GLOBAL_MIN_LINES_SHORT = 4
GLOBAL_MIN_LINES_LONG = 8


#
def twist_solo_diagonal(df):
  # Annotate solo diagonal twists
  twists = list(df['Twist angle'])
  is_diag = lambda twist_str: 'diagonal' in twist_str
  res = [True] if is_diag(twists[0]) else [False]
  for i in range(1, len(twists)):
    if is_diag(twists[i]) and twists[i-1] not in ['90', '180']:
      res.append(True)
    else:
      res.append(False)
  return res





#
funcs = {
  'Twist solo diagonal': twist_solo_diagonal,
}
annot_types = {}
for a in funcs:
  if a not in annot_types:
    annot_types[a] = bool