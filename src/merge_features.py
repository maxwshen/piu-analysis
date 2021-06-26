import _config, util
import os
import pandas as pd

inp_dir = _config.OUT_PLACE + 'd_annotate/'

def main():
  fns = [fn for fn in os.listdir(inp_dir) if '_features' in fn]

  timer = util.Timer(total=len(fns))
  mdf = pd.DataFrame()
  for fn in fns:
    df = pd.read_csv(inp_dir + fn, index_col=0)
    mdf = mdf.append(df)
    timer.update()
  
  mdf.to_csv(_config.OUT_PLACE + f'features.csv')
  return


if __name__ == '__main__':
  main()