import numpy as np

# Units: Seconds; [before note, after note]
perfect_windows = {
  'piu nj': [0.0416, 0.0832],
  'piu hj': [0.025, 0.0666],
  'piu vj': [0.0083, 0.0499],
  'ddr': [0.0167, 0.0167],
  'itg': [0.0215, 0.0215],
}

# min. level to propose brackets alongside jumps for two-hits (01100)
bracket_level_threshold = 15
# min. level to propose alternating feet in hold-taps with brackets
hold_bracket_level_threshold = 19

min_footswitch_level = 13

# max lines in hold-taps to apply penalty on alternating
hold_tap_line_threshold = 4

# Consider at most 4 lines for multihit. More can be proposed for incorrectly annotated stepcharts with very high BPM with notes
max_lines_in_multihit = 4

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
      'Angle too open': 0,
      'Angle duck': 0,
      'Angle extreme duck': 0,
      'Angle non-air inverted': 0,
      'Jump': 0.75,
      'Move power': 1.5,
      'No movement reward': -0.2,
      'Jacks': 0.0, 
      'Toe-heel alternate': 5,
      'Multi reward': -1.5,
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
      'Inverted feet small': 0.3,
      'Inverted feet big': 0.5,
      'Inverted hands': 5,
      'Hold footslide': 0.2,
      'Hold footswitch': 5,
      'Angle too open': 0,
      'Angle duck': 0,
      'Angle extreme duck': 0,
      'Angle non-air inverted': 0,
      'Jump': 0.75,
      'Move power': 1.5,
      'No movement reward': -.2,
      'Jacks': 0.0, 
      'Toe-heel alternate': 0.1,
      'Multi reward': -1.5,
      'Bracket': 0,
      'Bracket on 1panel line': 0,
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