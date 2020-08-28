import _config
import sys, os, fnmatch, datetime, subprocess
import numpy as np

# parameters

# Value = largest possible num. panels hit simultaneously
prev_panel_buffer_len = {
  'singles': 3,
  'doubles': 4,
}

perfect_windows = {
  'piu nj': [0.0416, 0.0832],
  'piu hj': [0.025, 0.0666],
  'piu vj': [0.0083, 0.0499],
  'ddr': [0.0167, 0.0167],
  'itg': [0.0215, 0.0215],
}

jacks_footswitch_npm_thresh = 275
jacks_footswitch_t_thresh = 60 / jacks_footswitch_npm_thresh

'''
  Movement costs
'''
movement_costs = {
  'beginner': {
    'Double step': 3,
    'Inverted feet small': 0.1,
    'Inverted feet big': 0.15,
    'Hold footslide': 1,
    'Hold footswitch': 3,
    'Angle too open': 3,
    'Angle duck': 1,
    'Angle extreme duck': 3,
    'Jump': 0.75,
    'No movement reward': -0.5,
    'Multi reward': 0,
    # Unused, since only air-X allowed
    'Bracket': 5,
    'Hands': 5,
    'Move without action': 3,
    'Downpress cost per limb': 0.05,

    # Distance of 1000 mm = 1 cost
    'Distance normalizer': 1000,
    'Inversion distance threshold': 200,
    'Time threshold': 0.5,
    # Time of 250 ms = 1 cost
    'Time normalizer': 0.25,
  },
  'basic': {
    'Double step': 3,
    'Inverted feet small': 0.1,
    'Inverted feet big': 0.15,
    'Hold footslide': 5,
    'Hold footswitch': 3,
    'Angle too open': 3,
    'Angle duck': 0.01,
    'Angle extreme duck': 3,
    'Jump': 0.75,
    'No movement reward': -0.5,
    # 'Multi reward': -10,
    'Multi reward': -1.5,
    'Bracket': 0,
    'Hands': 5,
    'Move without action': 3,
    'Downpress cost per limb': 0.05,

    # Distance of 1000 mm = 1 cost
    'Distance normalizer': 1000,
    'Inversion distance threshold': 200,
    'Time threshold': 0.75,
    # Time of 250 ms = 1 cost
    'Time normalizer': 0.25,
  },
  'advanced': {
    'Double step': 3,
    'Inverted feet small': 0.1,
    'Inverted feet big': 0.15,
    'Hold footslide': 5,
    'Hold footswitch': 3,
    'Angle too open': 3,
    'Angle duck': 0.01,
    'Angle extreme duck': 3,
    'Jump': 0.75,
    'No movement reward': -0.5,
    'Multi reward': --1.5,
    'Bracket': 0,
    'Hands': 5,
    'Move without action': 3,
    'Downpress cost per limb': 0.05,

    # Distance of 1000 mm = 1 cost
    'Distance normalizer': 1000,
    'Inversion distance threshold': 200,
    'Time threshold': 0.50,
    # Time of 200 ms = 1 cost
    'Time normalizer': 0.20,
  },
}