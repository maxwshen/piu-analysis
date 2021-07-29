import _config, util
import matplotlib as mpl
import matplotlib.pyplot as plt, pandas as pd, numpy as np, seaborn as sns
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, OffsetImage, AnnotationBbox
import matplotlib.ticker as plticker
import matplotlib.patches as patches
from PIL import Image
from adjustText import adjust_text
import functools

graphics_dir = _config.PRJ_DIR + 'graphics/'

pos_dfs = {
  'singles': pd.read_csv('../data/positions_singles.csv', index_col=0),
  'doubles': pd.read_csv('../data/positions_doubles.csv', index_col=0),
}

PAD_LENGTHS = {
  'singles': 838.2, # mm; 33 inches inner pad
  'doubles': 1714.5, # mm; 33 + 33 + 1.5 inch edge between pads
}
PAD_HEIGHT = 838.2
PAD_ALPHA = 0.3
FOOT_ALPHA = 0.2


'''
  Drawing
'''
class Artist():
  def __init__(self, singlesordoubles):
    assert singlesordoubles in ['singles', 'doubles']
    self.singlesordoubles = singlesordoubles

    if singlesordoubles == 'singles':
      self.ZOOM = 0.25
      self.CIRCLE_DIAMETER = 90
      self.FIG_BASE_SIZE = 5
      self.TEXT_SIZE = 25
      self.air_text_offset = (0, -20)
      self.pos_df = pos_dfs['singles']
    elif singlesordoubles == 'doubles':
      self.ZOOM = 0.5
      self.CIRCLE_DIAMETER = 80
      self.FIG_BASE_SIZE = 10
      self.TEXT_SIZE = 50
      self.air_text_offset = (0, -40)
      self.pos_df = pos_dfs['doubles']

    self.left_color = '#ec4339'
    self.right_color = '#00a0dc'
    self.left_arrow_color = '#e50000'
    self.right_arrow_color = '#0000ff'

    self.pos_names = set(self.pos_df['Name'])

    self.pad_img = mpimg.imread(graphics_dir + f'pad-{self.singlesordoubles}.PNG')
    self.foot_imgs = {
      'Left foot': self.load_foot_image('left-foot'),
      'Right foot': self.load_foot_image('right-foot'),
    }


  def load_foot_image(self, foot):
    im = Image.open(graphics_dir + f'{foot}.png')
    return im


  @functools.lru_cache(maxsize=None)
  def foot_pos_row(self, foot_pos):
    if foot_pos not in self.pos_names:
      print(f'Error: Failed to find {foot_pos}. Check singlesordoubles')
      raise Exception(f'Error: Failed to find {foot_pos}')
    return self.pos_df[self.pos_df['Name'] == foot_pos].iloc[0]


  @functools.lru_cache(maxsize=None)
  def get_loc_from_pos(self, pos, pos_type):
    row = self.foot_pos_row(pos)
    return (row[f'Loc x - {pos_type}'], -row[f'Loc y - {pos_type}'])


  def draw_pad(self):
    '''
      Draws pad image, returning (fig, ax)
      Data coordinates are in mm, with negative y: upper left is (0, 0)
    '''

    im = self.pad_img
    
    if self.singlesordoubles == 'singles':
      fig_dim = (self.FIG_BASE_SIZE, self.FIG_BASE_SIZE)
    else:
      doubles_to_singles_ratio = PAD_LENGTHS['doubles']/PAD_LENGTHS['singles']
      fig_dim = (self.FIG_BASE_SIZE*doubles_to_singles_ratio, self.FIG_BASE_SIZE)

    fig, ax = plt.subplots(figsize=fig_dim)
    
    pad_length = PAD_LENGTHS[self.singlesordoubles]
    ax.set_xlim(0, pad_length)
    ax.set_ylim(-PAD_HEIGHT, 0)
    ax.imshow(im, alpha=PAD_ALPHA, extent=[0, pad_length, -PAD_HEIGHT, 0])
    return fig, ax


  def draw_foot_from_pos(self, foot, foot_pos, ax):
    row = self.foot_pos_row(foot_pos)
    rotation = row['Rotation']
    center = (row['Loc x - center'], -1 * row['Loc y - center'])

    if 'a' not in foot_pos:
      im = self.foot_imgs[foot]
      im = im.rotate(-float(rotation), expand=True)
      im = np.array(im) / 255

      ax.add_artist(AnnotationBbox(
          OffsetImage(im, zoom=self.ZOOM, alpha=FOOT_ALPHA), center, frameon=False))
    else:
      # air, just draw circle
      color = self.left_color if foot == 'left-foot' else self.right_color
      circle = patches.Circle(
          center, self.CIRCLE_DIAMETER/2,
          linewidth=0, facecolor=color, alpha=FOOT_ALPHA)
      ax.add_patch(circle)
    return


  def draw_text(self, loc, text, ax):
    return ax.text(loc[0], loc[1], text,
        size=self.TEXT_SIZE, ha='center', va='center')


  def draw_arrows(self, poss, color, ax):
    # Drop first position and repeated positions
    if len(poss) == 0 or len(set(poss)) == 1:
      return

    first_pos = poss[0]
    # second_idx = min(i for i, pos in enumerate(poss) if pos != first_pos)
    second_idx = 0
    poss = drop_neighbor_duplicates(poss[second_idx:])
    
    for i in range(1, len(poss)):
      pos1, pos2 = poss[i-1], poss[i]
      
      foot_part = 'center'
      loc1 = self.get_loc_from_pos(pos1, foot_part)
      loc2 = self.get_loc_from_pos(pos2, foot_part)

      # Rescale arrow length    
      l1, l2 = np.array(loc1), np.array(loc2)
      vec_len = np.linalg.norm(l1-l2)
      min_vec_len = 1000
      ratio = max(1, min_vec_len / vec_len)
      loc1 = l1 + ratio*(l1-l2)/vec_len
      loc2 = l2 + ratio*(l2-l1)/vec_len
      
      offset = (0, 0)
      
      ax.arrow(loc1[0] + offset[0], loc1[1] + offset[1],
                loc2[0]-loc1[0], loc2[1]-loc1[1],
                head_length=30, head_width=40, overhang=0.2,
                facecolor=color, edgecolor=color,
                length_includes_head=True)
    return


  '''
    Primary
  '''
  def plot_single_sa(self, i, sa, texts, ax):
    [poss, actions] = sa.split(';')
    feet = ['Left foot', 'Right foot']
    for foot, pos, action in zip(feet, poss.split(','), actions.split(',')):
      self.draw_foot_from_pos(foot, pos, ax)
      
      pos_row = self.pos_df[self.pos_df['Name'] == pos].iloc[0]
      action_string = f'{i+1}'
      
      if action[0] in set(list('12')):
        loc = (pos_row['Loc x - heel ball'], -pos_row['Loc y - heel ball'])
        t = self.draw_text(loc, action_string, ax)
        texts.append(t)
          
      if action[1] in set(list('12')):
        loc = (pos_row['Loc x - toe ball'], -pos_row['Loc y - toe ball'])
        # Offset air-toe hit text
        loc = np.array(loc) + np.array(self.air_text_offset)
        t = self.draw_text(loc, action_string, ax)
        texts.append(t)

    return poss.split(',')


  def plot_sas(self, sas):
    fig, ax = self.draw_pad()
    
    left_poss, right_poss = [], []
    texts = []
    for i, sa in enumerate(sas):
      poss = self.plot_single_sa(i, sa, texts, ax)

      left_poss.append(poss[0])
      right_poss.append(poss[1])
    
    self.draw_arrows(left_poss, self.left_arrow_color, ax)
    self.draw_arrows(right_poss, self.right_arrow_color, ax)

    sns.despine(bottom=True, left=True)
    ax.axis('off')
    plt.tight_layout()

    adjust_text(texts)

    return fig


'''
  Helper
'''
def drop_neighbor_duplicates(items):
  res = [items[0]]
  for i in range(1, len(items)):
    if items[i] != items[i-1]:
      res.append(items[i])
  return res


def fig2img(fig):
  # Convert a Matplotlib figure to a PIL Image
  import io
  buf = io.BytesIO()
  fig.savefig(buf)
  buf.seek(0)
  img = Image.open(buf)
  return img


def plot_choreo_inset(singlesordoubles, sas):
  # Plots choreography, returns cropped image for insetting
  artist = Artist(singlesordoubles)
  fig = artist.plot_sas(sas)
  img = fig2img(fig)
  plt.close()

  if singlesordoubles == 'singles':
    xbound = 70
    ybound = 100
  elif singlesordoubles == 'doubles':
    xbound = 140
    ybound = 200

  width, height = img.size
  adj_width = width - 2*xbound
  adj_height = height - 2*ybound
  img = img.crop((xbound, ybound, width-xbound, height-ybound))
  return img, adj_width / adj_height


def plot_choreo(singlesordoubles, sas, out_fn):
  # does not crop
  artist = Artist(singlesordoubles)
  fig = artist.plot_sas(sas)
  fig.savefig(out_fn)
  return


def test():
  # test_sas = [
  #   '14,69;--,-1',
  #   '54,69;1-,--',
  #   '54,36;--,1-',
  #   '14,36;1-,--',
  # ]
  test_sas = [
    '47,36;-1,1-',
    '14,69;1-,-1',
    '47,36;-1,1-',
    '14,69;1-,-1',
  ]

  plot_choreo('singles', test_sas, 'test-choreo.png')
  return

if __name__ == '__main__':
  test()