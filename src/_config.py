import sys, os

SRC_DIR = os.path.dirname(os.path.realpath(__file__)) + '/'
PRJ_DIR = '/'.join(SRC_DIR.split('/')[:-2]) + '/'

#######################################################
# Note: Directories should end in / always
#######################################################
DATA_DIR = PRJ_DIR + 'data/'
OUT_PLACE = PRJ_DIR + 'out/'
RESULTS_PLACE = PRJ_DIR + 'results/'
QSUBS_DIR = PRJ_DIR + 'qsubs/'
#######################################################
#######################################################

# which data are we using? import that data's parameters
# DATA_FOLD = 'rename_me/'
DATA_FOLD = ''

sys.path.insert(0, DATA_DIR + DATA_FOLD)
import _dataconfig as d
print('Using data folder:\n', DATA_DIR + DATA_FOLD)
DATA_DIR += DATA_FOLD
OUT_PLACE += DATA_FOLD
RESULTS_PLACE += DATA_FOLD
QSUBS_DIR += DATA_FOLD
