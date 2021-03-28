import _config
import sys, os, fnmatch, datetime, subprocess
import numpy as np

# parameters

# Value = largest possible num. panels hit simultaneously
prev_panel_buffer_len = {
  'singles': 3,
  'doubles': 4,
}

# Units: Seconds; [before note, after note]
perfect_windows = {
  'piu nj': [0.0416, 0.0832],
  'piu hj': [0.025, 0.0666],
  'piu vj': [0.0083, 0.0499],
  'ddr': [0.0167, 0.0167],
  'itg': [0.0215, 0.0215],
}

jacks_footswitch_npm_thresh = 275
jacks_footswitch_t_thresh = 60 / jacks_footswitch_npm_thresh

# Consider at most 4 total lines for multihit. More can be proposed for incorrectly annotated stepcharts with very high BPM with notes
max_lines_in_multihit = 4

'''
  Movement costs
  TODO - Move to CSV?
'''
movement_costs = {
  'beginner': {
    'costs': {
      'Double step': 3,
      'Inverted feet small': 0.1,
      'Inverted feet big': 0.15,
      'Inverted hands': 5,
      'Hold footslide': 1,
      'Hold footswitch': 3,
      'Angle too open': 3,
      'Angle duck': 1,
      'Angle extreme duck': 3,
      'Angle non-air inverted': 1,
      'Jump': 0.75,
      'No movement reward': -0.5,
      'Multi reward': 0,
      # Unused, since only air-X allowed
      'Bracket': 5,
      'Hands': 5,
      'Move without action': 3,
      'Downpress cost per limb': 0.05,
    },
    'parameters': {
      # Distance of 1000 mm = 1 cost
      'Distance normalizer': 1000,
      'Inversion distance threshold': 200,
      'Time threshold': 0.5,
      # Time of 250 ms = 1 cost
      'Time normalizer': 0.25,
    },
  },
  'basic': {
    'costs': {
      'Double step': 3,
      'Inverted feet small': 0.1,
      'Inverted feet big': 0.15,
      'Inverted hands': 5,
      'Hold footslide': 0.2,
      'Hold footswitch': 3,
      'Angle too open': 3,
      'Angle duck': 0.01,
      'Angle extreme duck': 3,
      'Angle non-air inverted': 1,
      'Jump': 0.75,
      'No movement reward': -0.5,
      'Multi reward': -1.5,
      'Bracket': 0,
      'Hands': 5,
      'Move without action': 3,
      'Downpress cost per limb': 0.05,
    },
    'parameters': {
      # Distance of 1000 mm = 1 cost
      'Distance normalizer': 1000,
      'Inversion distance threshold': 200,
      'Time threshold': 0.50,
      # Time of 200 ms = 1 cost
      'Time normalizer': 0.20,
    },
  },
  'basicold': {
    'costs': {
      'Double step': 3,
      'Inverted feet small': 0.1,
      'Inverted feet big': 0.15,
      'Inverted hands': 5,
      'Hold footslide': 0.2,
      'Hold footswitch': 3,
      'Angle too open': 3,
      'Angle duck': 0.01,
      'Angle extreme duck': 3,
      'Angle non-air inverted': 1,
      'Jump': 0.75,
      'No movement reward': -0.5,
      # 'Multi reward': -10,
      'Multi reward': -1.5,
      'Bracket': 0,
      'Hands': 5,
      'Move without action': 3,
      'Downpress cost per limb': 0.05,
      'Min cost': 1.5,
    },
    'parameters': {
      # Distance of 1000 mm = 1 cost
      'Distance normalizer': 1000,
      'Inversion distance threshold': 200,
      'Time threshold': 0.75,
      # Time of 250 ms = 1 cost
      'Time normalizer': 0.25,
    },
  },
}