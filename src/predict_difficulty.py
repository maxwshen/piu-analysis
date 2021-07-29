'''
  Predicts charts difficulty
  Without a model of life loss, predictions are better interpreted as
    difficulty of S or SSS rather than stage passing
'''
import _config
import numpy as np, pandas as pd

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

all_df = pd.read_csv(_config.OUT_PLACE + 'a_format_data/all_stepcharts.csv', index_col=0)
all_df['Level'] = all_df['METER']

'''
  Rescale predictions
'''
def min_rescale(data, level):
  '''
    Rescale by minimum multiplicative factor that places >X% of data
      within -0.5 and +0.5 of level.
    Center data such that median is equal to level.
  '''
  freq_within_1_lower = 0.50
  freq_within_1_upper = 0.75
  lower_level = 10
  upper_level = 16

  data = np.array(data)
  centered_data = data - np.median(data)
  
  frac_in = lambda data: sum(bool(d>-0.5 and d<0.5) for d in data) / len(data)
  
  if level >= upper_level:
    freq_threshold = freq_within_1_upper
  elif level <= lower_level:
    freq_threshold = freq_within_1_lower
  else:
    fq_range = (freq_within_1_upper - freq_within_1_lower)
    rat = (level - lower_level) / (upper_level - lower_level)
    freq_threshold = freq_within_1_lower + fq_range * rat

  scale = 1
  scaled_data = centered_data * scale
  while frac_in(scaled_data) < freq_threshold:
    scale -= 0.05
    scaled_data = centered_data * scale
  
  scaled_data += level
  return list(scaled_data)


def rescale_preds_annotate(df):
  new_df = pd.DataFrame()
  for level in set(df['Level']):
    dfs = df[df['Level'] == level]
    preds = list(dfs['Predicted level unscaled'])
    rescaled_preds = min_rescale(preds, level)
    dfs.loc[:, 'Predicted level'] = rescaled_preds

    # Grouped by level: Annotate charts by percentile rating
    pctile = lambda p, data: sum(p>=d for d in data) / len(data)
    percentiles = [pctile(p, rescaled_preds) for p in rescaled_preds]
    dfs.loc[:, 'Predicted level percentile'] = percentiles
    new_df = new_df.append(dfs)

  new_df.loc[:, 'Residual'] = new_df['Predicted level'] - new_df['Level']
  return new_df


'''
  Modeling
'''
def get_monotonic_constraints(ft_cols):
  cst = []
  no_rel = ['Hold', 'Irregular rhythm', 'Rhythm change',
            'Twist angle - none', 'Stepchart']
  for col in ft_cols:
    if any(x in col for x in no_rel):
      cst.append(0)
    else:
      cst.append(1)
  return cst


def predict(df, ft_cols):
  nms = list(df['Name (unique)'])
  sctypes = []
  for typ in ['arcade', 'fullsong', 'shortcut', 'remix']:
    sctype = f'Stepchart - {typ}'
    sctypes.append(sctype)
    df.loc[:, sctype] = [int(bool(typ in nm)) for nm in nms]

  ft_cols += sctypes
  x = np.array(df[ft_cols])
  y = np.array(df['Level'])

  model = HistGradientBoostingRegressor(
    monotonic_cst=get_monotonic_constraints(ft_cols)
  )

  # Train on entire dataset, then rescale residuals
  model.fit(x, y)
  score = model.score(x, y)
  print(f'Score: {score}')
  if score < 0.8:
    raise Exception(f'Error: Fitted model score is too low')

  pred_level = model.predict(x)
  df['Predicted level unscaled'] = pred_level
  df['Residual unscaled'] = pred_level - y

  df = rescale_preds_annotate(df)
  return df


def test_local():
  df = pd.read_csv('../out/features.csv', index_col=0)
  df['Name (unique)'] = df.index
  ft_cols = [x for x in df.columns if x != 'Name (unique)']
  df = df.merge(all_df, on='Name (unique)', how='left')

  df['Is singles'] = (df['Steptype simple'].str.contains('S'))
  df['Is doubles'] = (df['Steptype simple'].str.contains('D'))

  print(f'Predicting singles difficulty ...')
  dfs = predict(df[df['Is singles']], ft_cols)

  print(f'Predicting doubles difficulty ...')
  dfd = predict(df[df['Is doubles']], ft_cols)

  mdf = dfs.append(dfd)
  mdf.to_csv('../out/merge_features/' + 'features.csv')
  return


if __name__ == '__main__':
  # Run local
  test_local()