import _config
import sys, os, fnmatch, datetime, subprocess
import numpy as np

# parameters

prev_panel_buffer_len = 4

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
  'Double step per limb': 1,
}