import _config
import os
import pandas as pd

inp_dir = _config.OUT_PLACE + 'd_annotate/'

def main():
  fns = [fn for fn in os.listdir(inp_dir) if 'features' in fn]

  mdf = pd.DataFrame()
  for fn in fns:
    df = pd.read_csv(inp_dir + fn, index_col=0)
    mdf = mdf.append(df)
  
  mdf.to_csv(inp_dir + f'features.csv')
  return


if __name__ == '__main__':
  main()