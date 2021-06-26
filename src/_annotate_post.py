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
  twists = df['Twist angle']
  res = [True] if twists[0] == 'diagonal' else [False]
  for i in range(1, len(twists)):
    if twists[i] == 'diagonal' and twists[i-1] not in ['90', '180']:
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