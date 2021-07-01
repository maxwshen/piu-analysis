'''
    Logic on stepchart info, e.g., get level from name
    Relies on stepchart df produced by a_format_data.py
'''
import _config
import pandas as pd

inp_dir = _config.OUT_PLACE + f'a_format_data/'

class SCInfo():
  def __init__(self):
    self.df = pd.read_csv(inp_dir + 'all_stepcharts.csv', index_col=0)

    self.name_to_level = {nm: lvl for nm, lvl in zip(self.df['Name (unique)'],
                                                     self.df['METER'])}

    pass

def singles_subset(df):
  return df[df['Steptype simple'].str.contains('S')]

def doubles_subset(df):
  return df[df['Steptype simple'].str.contains('D')]