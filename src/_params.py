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

'''
  Movement costs
'''
movement_costs = {
  'default': {
    'Double step per limb': 2,
    'Bracket': 0.5,
    'Hands': 5,
    'Inverted feet': 1,
    'Hold footslide': 1,
    'Hold footswitch': 3,
    'Angle too open': 3,
    'Angle duck': 1,
    'Angle extreme duck': 3,
    'Jump': 0.5,

    # Distance of 1000 mm = 1 cost
    'Distance normalizer': 1000,
    'Time threshold': 0.5,
    # Time of 250 ms = 1 cost
    'Time normalizer': 0.25,
    'Time forgive double step': 2,
  },
}