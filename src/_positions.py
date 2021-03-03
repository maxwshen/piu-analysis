#
import _config
import pandas as pd

singles_pos_df = pd.read_csv(_config.DATA_DIR + f'positions_singles.csv', index_col=0)
doubles_pos_df = pd.read_csv(_config.DATA_DIR + f'positions_doubles.csv', index_col=0)