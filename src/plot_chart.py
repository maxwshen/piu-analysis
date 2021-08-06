'''
  matplotlib-based plotting of chart
  mostly deprecated - front-end uses javascript instead
'''

import _config, util
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt, pandas as pd, numpy as np, seaborn as sns
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, OffsetImage, AnnotationBbox
from PIL import Image, ImageEnhance
from PIL import Image
import functools
from collections import defaultdict

import plot_choreo

graphics_dir = _config.PRJ_DIR + 'graphics/'

'''
  Parameters
'''
ZOOM = 1.75
HEIGHT_PER_LINE = 0.4
CHOREO_ZOOM = 0.55

CHOREO_X_LEFT = -1
CHOREO_X_RIGHT = 1

ANNOT_X_LEFT = 0
ANNOT_X_RIGHT = 1

HOLD_ALPHA = 0.3
HOLD_WIDTH = 0.6 # relative to 1.0

ARROW_ALPHA_TOE = 0.6
ARROW_ALPHA_HEEL = 0.8
ARROW_ALPHA_HAND = 0.9

CHART_GRID_COLORS = ['blue', 'red', 'orange', 'red', 'blue']*2
CHART_GRID_ALPHA = 0.3


class Artist():
  def __init__(self, singlesordoubles):
    self.singlesordoubles = singlesordoubles
    if singlesordoubles == 'singles':
      self.figwidth = 7
      self.panels = ['p1,1', 'p1,7', 'p1,5', 'p1,9', 'p1,3']
    elif singlesordoubles == 'doubles':
      self.figwidth = 14
      self.panels = ['p1,1', 'p1,7', 'p1,5', 'p1,9', 'p1,3',
                     'p2,1', 'p2,7', 'p2,5', 'p2,9', 'p2,3']
    self.arrow_to_img = self.init_arrows()

    # [text, arrows, choreo]
    self.width_ratios = {
      'singles': [1, 1.5, 1.5],
      'doubles': [2, 3, 3],
    }

    self.panel_to_xloc = {p: i for i, p in enumerate(self.panels)}
    self.num_panels = len(self.panel_to_xloc)

    self.left_color = '#ec4339'
    self.right_color = '#00a0dc'
    self.hand_color = '#595c5f'

    '''
      want to color every 4th line differently, but hmm segmentation and major grid make it challenging to guarantee landing on the downbeat
    '''
    # self.major_grid_color = '#595c5f'
    self.major_grid_color = '#b6b9bc'
    self.major_grid_alpha = 0.5
    self.minor_grid_color = '#b6b9bc'
    self.minor_grid_alpha = 0.5

    self.limb_to_hold_color = {
      'Left foot': self.left_color,
      'Right foot': self.right_color,
      'Left hand': self.hand_color,
      'Right hand': self.hand_color,
    }


  def init_arrows(self):
    img_fold = graphics_dir + 'pixel/transparent-bg/'

    arrow_to_img_fn = {
      'Left foot p1,1':  f'{img_fold}/leftfoot-downleft.png',
      'Left foot p1,3':  f'{img_fold}/leftfoot-downright.png',
      'Left foot p1,5':  f'{img_fold}/leftfoot-center.png',
      'Left foot p1,7':  f'{img_fold}/leftfoot-upleft.png',
      'Left foot p1,9':  f'{img_fold}/leftfoot-upright.png',
      'Right foot p1,1': f'{img_fold}/rightfoot-downleft.png',
      'Right foot p1,3': f'{img_fold}/rightfoot-downright.png',
      'Right foot p1,5': f'{img_fold}/rightfoot-center.png',
      'Right foot p1,7': f'{img_fold}/rightfoot-upleft.png',
      'Right foot p1,9': f'{img_fold}/rightfoot-upright.png',
      'Left foot p2,1':  f'{img_fold}/leftfoot-downleft.png',
      'Left foot p2,3':  f'{img_fold}/leftfoot-downright.png',
      'Left foot p2,5':  f'{img_fold}/leftfoot-center.png',
      'Left foot p2,7':  f'{img_fold}/leftfoot-upleft.png',
      'Left foot p2,9':  f'{img_fold}/leftfoot-upright.png',
      'Right foot p2,1': f'{img_fold}/rightfoot-downleft.png',
      'Right foot p2,3': f'{img_fold}/rightfoot-downright.png',
      'Right foot p2,5': f'{img_fold}/rightfoot-center.png',
      'Right foot p2,7': f'{img_fold}/rightfoot-upleft.png',
      'Right foot p2,9': f'{img_fold}/rightfoot-upright.png',
    }

    # Modify brightnesses
    brightnesses = {'heel': 0.95, 'toe':  1.35}
    arrow_to_img = {}
    for part, brightness in brightnesses.items():
      d = {}
      for arrow, fn in arrow_to_img_fn.items():
        im = Image.open(fn)
        enhancer = ImageEnhance.Brightness(im)
        im_output = enhancer.enhance(brightness)
        d[arrow] = np.array(im_output) / 255
      arrow_to_img[part] = d

    # Add grayscale images for hands
    def rgba_to_gray(im):
      # rgba image: (n, m, 4)
      im = np.array(im)
      [n, m, _] = im.shape
      arr = np.array(im)
      gs = np.mean(arr[:,:,:3], -1)
      gsr = np.reshape(gs, (n, m, 1))
      gs3 = np.tile(gsr, 3)
      alpha = np.reshape(im[:,:,-1], (n, m, 1))
      return np.concatenate([gs3, alpha], -1)

    hand_arrow_to_img = dict()
    for arrow, fn in arrow_to_img_fn.items():
      im = Image.open(fn)
      im_output = rgba_to_gray(im)
      hand_arrow = arrow.replace('foot', 'hand')
      hand_arrow_to_img[hand_arrow] = np.array(im_output) / 255
    
    arrow_to_img['hand'] = hand_arrow_to_img
    return arrow_to_img


  '''
    Drawing
  '''
  def draw_hold(self, limb, panel, start_beat, end_beat, axes):
    xloc = self.panel_to_xloc[panel]
    color = self.limb_to_hold_color[limb]
    for ax in axes:
      rect = mpl.patches.Rectangle(
          (xloc - HOLD_WIDTH/2, start_beat), 
          HOLD_WIDTH, end_beat - start_beat,
          linewidth=1, facecolor=color, alpha=HOLD_ALPHA)
      ax.add_patch(rect)
    return


  def draw_arrow_sa(self, limb, panel, beat, sa, axes):
    d = parse_sa_to_text(sa)

    xloc = self.panel_to_xloc[panel]
    yloc = beat
    if limb == 'Left foot':
      text = 'L'
    elif limb == 'Right foot':
      text = 'R'

    # find among ['heel 5', 'toe 7'] which one belongs to panel 'p1,7'
    action = [x for x in d[limb] if x[-1] == panel[-1]][0]
    if 'foot' in limb:
      part = ''
      if 'heel' in action:
        text += 'H'
        alpha = ARROW_ALPHA_HEEL
        part = 'heel'
      elif 'toe' in action:
        text += 'T'
        alpha = ARROW_ALPHA_TOE
        part = 'toe'
    else:
      part = 'hand'
      text = ''
      alpha = ARROW_ALPHA_HAND
    
    im = self.arrow_to_img[part][f'{limb} {panel}']
    
    text_color = 'black'
    for ax in axes:
      ax.add_artist(AnnotationBbox(OffsetImage(im, zoom=ZOOM, alpha=alpha),
                                   (xloc, yloc), frameon=False))
      ax.add_artist(AnnotationBbox(TextArea(text, textprops=dict(color=text_color)),
                                   (xloc, yloc), frameon=False))
      pass
    return


  '''
    Primary
  '''
  def plot_section(self, df, section, out_fn):
    start, end = section
    stats = get_section_stats(section, df)
    dfs = df.iloc[start:end]

    min_time = min(dfs['Time'])
    max_time = max(dfs['Time'])
    ytick_minor_interval = stats['Median time since downpress']
    ytick_major_interval = 4 * ytick_minor_interval

    num_minor_yticks = round((max_time - min_time) / ytick_minor_interval)
    num_buffer_lines = 5
    num_lines = num_minor_yticks + num_buffer_lines
    figheight = num_lines * HEIGHT_PER_LINE
    wr = self.width_ratios[self.singlesordoubles]
    fig, axes = plt.subplots(1, 3,
                            figsize=(self.figwidth, figheight), 
                            gridspec_kw={'width_ratios': wr}, 
                            sharey=True)
    annot_axes = [axes[0]]
    arrow_axes = [axes[1]]
    choreo_axes = [axes[2]]

    arrow_axes[0].set_ylim(min_time - ytick_minor_interval,
                           max_time + 4*ytick_minor_interval)
    arrow_axes[0].invert_yaxis()


    # Plot arrows
    limbs = ['Left foot', 'Right foot', 'Left hand', 'Right hand']
    note_types = ['1', '2', '3', '4']
    active_holds = {}
    for i, row in dfs.iterrows():
      beat = row['Time']
      for limb in limbs:
        for note in note_types:
          col = f'{limb} {note}'
          if type(row[col]) != str:
            continue

          panels = row[col].split(';')
          if note in ['1', '2']:
            for p in panels:
              self.draw_arrow_sa(limb, p, beat, row['Stance action'], arrow_axes)
          if note == '2':
            for p in panels:
              active_holds[p] = (limb, beat)
          if note == '3':
            for p in panels:
              if p in active_holds:
                hold_limb, start_beat = active_holds[p]
                self.draw_hold(limb, p, start_beat, beat, arrow_axes)
                del active_holds[p]
          if note == '4':
            for p in panels:
              if p in active_holds:
                hold_limb, start_beat = active_holds[p]
                # Handle hold footswitches
                if hold_limb != limb:
                  self.draw_hold(hold_limb, p, start_beat, beat, arrow_axes)
                  active_holds[p] = (limb, beat)

    # Plot choreo
    round_down = lambda x, p: int(x * 10**p) / 10**p
    rounded_yti = round_down(ytick_major_interval, 4)
    for t in np.arange(min_time, max_time, rounded_yti):
      offset = ytick_minor_interval / 2
      int_start = t - offset
      int_end = t + rounded_yti * 0.95 - offset
      
      crit = (dfs['Time'] >= int_start) & (dfs['Time'] < int_end) & \
             (dfs['Has downpress'])
      choreo_dfs = dfs[crit]
      sas = list(choreo_dfs['Stance action'])
      
      choreo_im, hw_ratio = plot_choreo.plot_choreo_inset(self.singlesordoubles, sas)
      choreo_im = np.array(choreo_im) / 255

      x_extent = hw_ratio*(int_end - int_start)
      choreo_axes[0].imshow(choreo_im, extent=[0, x_extent, int_start, int_end],
                            origin='lower',
                            aspect='equal',
                            )

    # Plot text annotations
    for i, row in dfs.iterrows():
      if row['Has downpress']:
        time = row['Time']
        annots = get_top_annots_in_row(row, 2)
        xs = [1, 0]
        has = ['right', 'left']
        for x, text, ha in zip(xs, annots, has):
          annot_axes[0].text(x, time, text,
                              ha=ha, va='center')


    arrow_axes[0].yaxis.set_minor_locator(
        mpl.ticker.MultipleLocator(base=ytick_minor_interval))
    arrow_axes[0].yaxis.set_major_locator(
        mpl.ticker.MultipleLocator(base=ytick_major_interval))


    # Formatting
    for ax in arrow_axes:
      ax.tick_params('both', width=0)
      ax.set_xlim(left=-0.5, right=len(self.panel_to_xloc)-1++0.5)
      ax.set_xticks(list(self.panel_to_xloc.values()))
      ax.set_xticklabels([])
      ax.grid(which='major', linestyle='-',
              color=self.major_grid_color, alpha=self.major_grid_alpha)
      ax.grid(which='minor', linestyle='-',
              color=self.minor_grid_color, alpha=self.minor_grid_alpha)
      ax.set_ylabel('Time (seconds)')

      ax.yaxis.set_tick_params(labelleft=True)
      ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

      gridlines = ax.get_xgridlines()
      for gc, gl in zip(CHART_GRID_COLORS, gridlines):
        gl.set_color(gc)
        gl.set_alpha(CHART_GRID_ALPHA)
      ax.spines['right'].set_visible(False)
      ax.spines['left'].set_visible(False)
        
    for ax in choreo_axes:
      ax.set_xlim(left=0, right=x_extent)
      ax.axis('off')

    for ax in annot_axes:
      ax.set_xlim(left=ANNOT_X_LEFT, right=ANNOT_X_RIGHT)
      ax.axis('off')
    
    
    sns.despine(bottom=True, left=True)
    fig.tight_layout()
    # plt.subplots_adjust(wspace=0)
    fig.patch.set_facecolor('white')
    fig.savefig(f'{out_fn}', bbox_inches='tight')
    plt.close()
    return


'''
  Plotting instructions for js
'''
def js_arrows(line_dfs, stance):
  '''
    arrows: list, elements: [x, y, 'LT/LH/RH/RT'],
    holds: list, elements: [x, y_start, y_end, 'LT/LH/RH/RT'],
    
    js: draw arrows using x, color by LT/LH/RH/RT or x
  '''
  panels = ['p1,1', 'p1,7', 'p1,5', 'p1,9', 'p1,3',
            'p2,1', 'p2,7', 'p2,5', 'p2,9', 'p2,3']
  panel_to_xloc = {p: i for i, p in enumerate(panels)}

  limbs = ['Left foot', 'Right foot', 'Left hand', 'Right hand']
  note_types = ['1', '2', '3', '4']
  active_holds = {}
  arrows, holds = [], []
  for i, row in line_dfs.iterrows():
    time = row['Time']
    for limb in limbs:
      for note in note_types:
        col = f'{limb} {note}'
        if type(row[col]) != str:
          continue

        panels = row[col].split(';')
        if note in ['1', '2']:
          for p in panels:
            x = panel_to_xloc[p]
            arrow_text = stance.get_limb_part_text(p, row['Stance action'], limb)
            arrows.append([x, float(time), arrow_text])
        if note == '2':
          for p in panels:
            arrow_text = stance.get_limb_part_text(p, row['Stance action'], limb)
            active_holds[p] = (arrow_text, time)
        if note == '3':
          for p in panels:
            if p in active_holds:
              arrow_text, start_time = active_holds[p]
              x = panel_to_xloc[p]
              holds.append([x, float(start_time), float(time), arrow_text])
              del active_holds[p]
        if note == '4':
          for p in panels:
            if p in active_holds:
              prev_arrow_text, start_time = active_holds[p]
              x = panel_to_xloc[p]
              arrow_text = stance.get_limb_part_text(p, row['Stance action'], limb)
              # Handle hold footswitches
              if prev_arrow_text != arrow_text:
                holds.append([x, float(start_time), float(time), prev_arrow_text])
                active_holds[p] = (arrow_text, time)
  return arrows, holds


def js_line_annotations(line_dfs):
  times, annotations = [], []
  for i, row in line_dfs.iterrows():
    if row['Has downpress']:
      time = row['Time']
      annots = get_top_annots_in_row(row, 2)
      if annots:
        annotations.append(', '.join(annots))
        times.append(float(time))
  return times, annotations


def get_long_holds(min_time, max_time, all_holds):
  # detect long holds that should be plotted in section, trim boundaries
  long_holds = []
  for hold in all_holds:
    [x, start, end, text] = hold
    if (start < min_time and end > min_time) or \
       (start < max_time and end > max_time):
      long_holds.append([x, max(start, min_time), min(end, max_time), text])
  return long_holds


'''
  Misc.
'''
def get_section_stats(section, df):
  dfs = df.iloc[section[0]:section[1]]
  dp_dfs = dfs[dfs['Has downpress']]

  first_beat, last_beat = min(dp_dfs['Beat']), max(dp_dfs['Beat'])    
  dfs = df[(df['Beat'] >= first_beat) & (df['Beat'] <= last_beat)]
  
  bs = dp_dfs['Beat since'][1:]
  bsd = dp_dfs['Beat since downpress'][1:]
  ts = dp_dfs['Time since downpress'][1:]
  median_bpm = np.median([bpm for bpm in dfs['BPM'] if bpm < 999])
  stats = {
    'First beat': min(dfs['Beat']),
    'Last beat': max(dfs['Beat']),
    'First time': min(dfs['Time']),
    'Last time': max(dfs['Time']),
    'Time length': max(dfs['Time']) - min(dfs['Time']),
    'Median beat since': np.median(bs),
    'Median beat since downpress': np.median(bsd),
    'Median time since downpress': np.median(ts),
    'Median bpm': median_bpm,
    'Beat time inc from bpm': 60 / (median_bpm),
    'Num. lines': len(dfs),
    'Num. downpress lines': len(dp_dfs),
  }
  
  stats['Median nps'] = (stats['Median bpm'] / stats['Median beat since downpress']) / 60
  
  '''
    - Calculate total figure height and size ratios from comparing num. lines across sections
    - Calculate ytick frequency from median time since
    
    Compare median time since to beat time inc from bpm, if we find a hit, report "16th notes @ X BPM"
    otherwise, just report "X BPM".
    
    Get nps color indicator from median time since
  '''
  
  return stats


def parse_sa_to_text(sa):
  [poss, actions] = sa.split(';')
  limbs = ['Left foot', 'Right foot', 'Left hand', 'Right hand'][:len(poss)]
  res = defaultdict(list)
  hits = set(list('12'))
  for limb, pos, action in zip(limbs, poss.split(','), actions.split(',')):
    for i, part in enumerate(['heel', 'toe']):
      if action[i] in hits:
        res[limb].append(f'{part} {pos[i]}')
  return res


'''
  Single line annotations
  Priority ordered
  Annotations not in this dict will not appear
'''
annots = {
  'Hands': 'Hands',
  'Staggered hit': 'Rolling hit',
  'Splits': 'Splits',
  'Spin': 'Spin',
  'Hold tap single foot': 'Bracket hold tap',
  'Hold footslide': 'Hold footslide',
  'Hold footswitch': 'Hold footswitch',
  'Footswitch': 'Footswitch',
  'Jack': 'Jack',
  'Double step': 'Double step',
  'Triple': 'Triple',
  'Quad': 'Quad',
  'Bracket': 'Bracket',
  'Broken stairs, doubles': '9 stair',
  'Stairs, doubles': '10 stair',
  'Stairs, singles': '5 stair',
  'Twist angle - 180': '180° twist',
  # 'Twist solo diagonal': 'Solo diag. twist',
  'Twist angle - far diagonal': 'Diag. twist',
  'Twist angle - close diagonal': 'Diag. twist',
  'Twist angle - 90': '90° twist',
}

ordered_annots = list(annots.keys())
annot_score = lambda x: ordered_annots.index(x)

def get_top_annots_in_row(row, n):
  res = [a for a in annots if a in row.index and row[a]]

  # twist_angle = row['Twist angle']
  # res.append(f'Twist angle - {twist_angle}')
  # res = [a for a in res if a in annots]

  top_annots = sorted(res, key=annot_score)[:n]
  return [annots[a] for a in top_annots]


'''
  Testing
'''
@util.time_dec
def test():
  # df = pd.read_csv('../out/d_annotate/Super Fantasy - SHK S16 arcade.csv')
  # artist = Artist('singles')
  # artist.plot_section(df, (15, 32), 'test-superfantasy-1.png')
  # artist.plot_section(df, (15, 60), 'test-superfantasy-2.png')

  # df = pd.read_csv('../out/d_annotate/Mitotsudaira - ETIA. D19 arcade.csv')
  # artist = Artist('doubles')
  # artist.plot_section(df, (56, 80), 'test-mitotsu-1.png')
  # artist.plot_section(df, (56, 100), 'test-mitotsu-2.png')

  df = pd.read_csv('../out/d_annotate/King of Sales - Norazo S21 arcade.csv')
  artist = Artist('singles')
  # artist.plot_section(df, (100, 159), '../out/tag_sections/test-kos-hands.png')
  artist.plot_section(df, (100, 110), '../out/tag_sections/test-kos-hands-short.png')
  return

if __name__ == '__main__':
  test()