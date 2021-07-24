import numpy as np

# Units: Seconds; [before note, after note]
perfect_windows = {
  'piu nj': [0.0416, 0.0832],
  'piu hj': [0.025, 0.0666],
  'piu vj': [0.0083, 0.0499],
  'ddr': [0.0167, 0.0167],
  'itg': [0.0215, 0.0215],
}

# Seconds
# want to include:
# faep x2 d24's rolling brackets are 58.3 ms apart
# some of windmill d23's rolling brackets are 61.0 ms apart
# full window is 83.2 ms (piu nj perfect)
multi_window = 0.062

# min. level to propose brackets alongside jumps for two-hits (01100)
bracket_level_threshold = 15
# min. level to propose alternating feet in hold-taps with brackets
hold_bracket_level_threshold = 19

min_footswitch_level = 13

# Min. number of lines to keep a multi drill
min_bracket_footswitch_drill_lines = 6

# max lines in hold-taps to apply penalty on alternating
hold_tap_line_threshold = 4

# Consider this many lines at most for multihit / rolling hits. 
max_lines_in_multihit = 2

init_stanceaction = {
  'singles': '14,36;--,--',
  'doubles': 'p1`36,p2`14;--,--',
}

'''
  Movement costs. TODO - Move to CSV?
'''
movement_costs = {
  'beginner': {
    'costs': {
      'Double step': 2.5,
      'Inverted feet small': 0.3,
      'Inverted feet big': 5,
      'Inverted hands': 5,
      'Hold footslide': 5,
      'Hold footswitch': 5,
      'Angle too open': 0.1,
      'Angle duck': 0.1,
      'Angle extreme duck': 0.5,
      'Angle non-air inverted': 1,
      'Jump': 0.75,
      'Move power': 1.5,
      'No movement reward': -0.2,
      'Jacks': 0.0, 
      'Toe-heel alternate': 5,
      'Multi reward': 0,
      'Double step in multi': 5,
      # Unused, since only air-X allowed
      'Bracket': 5,
      'Bracket on 1panel line': 5,
      'Hands': 5,
      'Move without action': 5,
      'Hold alternate feet for hits (onetime, short)': 5,
      'Hold alternate feet for hits (onetime, long)': 1,
      'Hold free feet for hits (onetime, short)': 10,
      'Hold free feet for hits (onetime, long)': 2,
      'Downpress cost per limb': 0,
    },
    'parameters': {
      # Distance of 1000 mm = 1 cost
      'Distance normalizer': 1000,
      'Inversion distance threshold': 250,
      'Time threshold': 5,
      # Time of 250 ms = 1 cost
      'Time normalizer': 0.25,
    },
  },
  'basic': {
    'costs': {
      'Double step': 2.5,
      'Inverted feet small': 1.5,
      'Inverted feet big': 2.25,
      'Inverted hands': 5,
      'Hold footslide': 0.2,
      'Hold footswitch': 10,
      'Angle too open': 0.1,
      'Angle duck': 0.1,
      'Angle extreme duck': 0.5,
      'Angle non-air inverted': 1,
      'Jump': 0.75,
      'Move power': 1.5,
      'No movement reward': -.2,
      'Jacks': 0.0, 
      'Toe-heel alternate': 0.1,
      'Multi reward': -5.0,
      'Double step in multi': 5,
      'Bracket': 0.05,
      'Bracket on 1panel line': 0.1,
      'Hands': 5,
      'Move without action': 5,
      'Hold alternate feet for hits (onetime, short)': 5,
      'Hold alternate feet for hits (onetime, long)': 1,
      'Hold free feet for hits (onetime, short)': 10,
      'Hold free feet for hits (onetime, long)': 2,
      'Downpress cost per limb': 0,
    },
    'parameters': {
      # Distance in mm = 1 cost
      'Distance normalizer': 250,
      'Inversion distance threshold': 250,
      'Time threshold': 0.50,
      # Time of 200 ms = 1 cost
      'Time normalizer': 0.20,
    },
  },
}

bracketable_lines = set([
  '10100', '01100', '00110', '00101',
  '1010000000',
  '0110000000',
  '0011000000',
  '0010100000',
  '0000010100',
  '0000001100',
  '0000000110',
  '0000000101',
  '0000110000',
  '0001001000',
])