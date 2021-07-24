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

    self.singles_nms = set(singles_subset(self.df)['Name (unique)'])
    self.doubles_nms = set(doubles_subset(self.df)['Name (unique)'])
    def name_to_sord(nm):
      if nm in self.singles_nms:
        return 'singles'
      if nm in self.doubles_nms:
        return 'doubles'
      raise Exception('Error: Name not found in singles or doubles')
    self.name_to_singleordouble = name_to_sord

def singles_subset(df):
  return df[df['Steptype simple'].str.contains('S')]

def doubles_subset(df):
  return df[df['Steptype simple'].str.contains('D')]