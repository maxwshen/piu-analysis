'''
    Logic re note lines from .ssc file
'''
import re, functools
import _params

def has_downpress(line):
  return num_downpress(line) > 0


def num_downpress(line):
  return sum(line.count(x) for x in list('12'))


def num_pressed(line):
  return sum(line.count(x) for x in list('124'))


def has_notes(line):
  if '`' in line:
    return bool(set(line) != set(['`0']))
  else:
    return bool(set(line) != set(['0']))


def is_hold_release(line):
  if '`' in line:
    return bool(set(line) != set(['`03']))
  else:
    return bool(set(line) != set(['03']))


def has_active_hold(line):
    return '4' in line


def frac_bracketable(lines):
  tlines = [x.replace('2', '1') for x in lines]
  num_bracketable = sum([l in _params.bracketable_lines for l in tlines])
  return num_bracketable / len(lines)


def add_active_holds(line, active_holds, panel_to_idx):
  # Add active holds into line as '4'
  # 01000 -> 01040
  aug_line = list(line)
  for panel in active_holds:
    idx = panel_to_idx[panel]
    if aug_line[idx] == '0':
      aug_line[idx] = '4'
  return ''.join(aug_line)

@functools.lru_cache(maxsize=None)
def parse_line(line):
  '''
    Handle lines like:
      0000F00000
      00{2|n|1|0}0000000    
      0000{M|n|1|0} -> 0
  '''
  ws = re.split('{|}', line)
  nl = ''
  for w in ws:
    if '|' not in w:
      nl += w
    else:
      nl += w[0]
  line = nl

  replace = {
    'F': '1',
    'M': '0',
    '4': '2',
    '6': '2',
  }
  line = line.translate(str.maketrans(replace))
  return line
