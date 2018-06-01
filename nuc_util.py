import random, sys, string
#import uuid


# #   Globals  # #

QUIET            = False # Global verbosity flag
LOGGING          = False # Global file logging flag
#TEMP_ID          = '%s' % uuid.uuid4()
#LOG_FILE_PATH    = 'nuc-tools-out-%s.log' % TEMP_ID
LOG_FILE_OBJ     = None # Created when needed

import nuc_parallel as parallel
import nuc_io as io


# #   Srcreen reporting  # # 

def report(msg):
 
  global LOG_FILE_OBJ
  
  if LOGGING:
    if not LOG_FILE_OBJ:
      LOG_FILE_OBJ = open(LOG_FILE_PATH, 'w')
      
    LOG_FILE_OBJ.write(msg)
  
  if not QUIET:
    print(msg)
   

def warn(msg, prefix='WARNING'):

  report('%s: %s' % (prefix, msg))

 
def critical(msg, prefix='FAILURE'):

  report('%s: %s' % (prefix, msg))
  sys.exit(0)


def info(msg, prefix='INFO'):

  report('%s: %s' % (prefix, msg))


# #  Strings  # #

 
def get_rand_string(size):
  
  return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for i in range(size))



