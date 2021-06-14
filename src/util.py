# Utility library functions: IO, OS stuff

import sys, string, csv, os, fnmatch, datetime, subprocess

#########################################
# TIME
#########################################

class Timer:
  def __init__(self, total = -1, print_interval = 20000):
    # print_interval is in units of microseconds
    self.times = [datetime.datetime.now()]
    self.num = 0
    self.last_print = 0
    self.prev_num = 0
    self.total = int(total)
    if sys.stdout.isatty():   # if undirected stdout
      self.print_interval = print_interval
    else:   # else if stdout is directed int ofile
      self.print_interval = 5000000   # 5 seconds

  def progress_update(self, done=False):
    if self.last_print == 0:
      num_secs = (datetime.datetime.now() - self.times[0]).microseconds
    else:
      num_secs = (datetime.datetime.now() - self.last_print).microseconds

    passed_print_interval = (num_secs >= self.print_interval)
    is_done = bool(self.num == self.total) or done

    if passed_print_interval or is_done:
      if self.last_print != 0:
        sys.stdout.write("\033[F\033[F\033[F\033[F\033[F")
      self.last_print = datetime.datetime.now()
      if self.total != -1:
        print('\n\t\tPROGRESS %:', '{:5.2f}'.format(float(self.num * 100) / float(self.total)), ' : ', self.num, '/', self.total)
        print('\t\t', self.progress_bar(float(self.num * 100) / float(self.total)))
      else:
        print('\n\t\tTIMER:', self.num, 'iterations done after', str(datetime.datetime.now() - self.times[0]), '\n')
        print()
      rate = float(self.num - self.prev_num) / num_secs
      a = (self.times[1] - self.times[0]) / self.num
      if rate > 1:
        print('\t\t\tRate:', '{:5.2f}'.format(rate), 'iterations/second')
      else:
        print('\t\t\tAvg. Iteration Time:', a)
        
      if self.total != -1:
        if not is_done:
          print('\t\tTIMER ETA:', a * self.total - (datetime.datetime.now() - self.times[0]))
        if is_done:
          print('\t\tCompleted in:', datetime.datetime.now() - self.times[0])

      self.prev_num = self.num

      sys.stdout.flush()

    sys.stdout.flush()
    return

  def update(self, print_progress = True):
    if len(self.times) < 2:
      self.times.append(datetime.datetime.now())
    else:
      self.times[-1] = datetime.datetime.now()
    self.num += 1

    if print_progress:
      self.progress_update()
    return

  def end(self):
    self.progress_update(done=True)
    return

  def progress_bar(self, pct):
    RESOLUTION = 40
    bar = '['
    pct = int(pct / (100.0 / RESOLUTION))
    bar += '\x1b[6;30;42m'
    for i in range(pct):
      bar += 'X'
    bar += '\x1b[0m'      
    for i in range(RESOLUTION - pct):
      bar += '-'
    bar += ']'
    return bar

# end Timer

def time_dec(func):
  def wrapper(*args, **kwargs):
    t = datetime.datetime.now()
    print('\n', t)
    res = func(*args, **kwargs)
    print(datetime.datetime.now())
    print('Completed in', datetime.datetime.now() - t, '\n')
    return res
  return wrapper

#########################################
# OS
#########################################
def ensure_dir_exists(directory):
  # Guarantees that input dir exists
  if not os.path.exists(directory):
    try:
      os.makedirs(directory)
    except OSError:
      if not os.path.isdir(directory):
        raise
  return

def exists_empty_fn(fn):
  ensure_dir_exists(os.path.dirname(fn))
  with open(fn, 'w') as f:
    pass
  return

def get_fn(string):
  # In: Filename (possibly with directories)
  # Out: Filename without extensions or directories
  return string.split('/')[-1].split('.')[0]

def line_count(fn):
  try:
    ans = subprocess.check_output(['wc', '-l', fn.strip()])
    ans = int(ans.split()[0])
  except OSError as err:
    print('OS ERROR:', err)
  return ans
