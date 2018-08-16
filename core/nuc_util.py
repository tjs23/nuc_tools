import random, sys, string, subprocess
import uuid
import numpy as np

# #   Globals  # #

QUIET            = False # Global verbosity flag
LOGGING          = False # Global file logging flag
TEMP_ID          = '%s' % uuid.uuid4()

LOG_FILE_PATH    = 'nuc-tools-out-%s.log' % TEMP_ID
LOG_FILE_OBJ     = None # Created when needed

import nuc_parallel as parallel

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


# #  Run  # # 
     
def call(cmd_args, stdin=None, stdout=None, stderr=None, verbose=True, wait=True, path=None, shell=False):
  """
  Wrapper for external calls to log and report commands,
  open stdin, stderr and stdout etc.
  """
  
  if verbose:
    info(' '.join(cmd_args))
  
  if path:
    env = dict(os.environ)
    prev = env.get('PATH', '')
    
    if path not in prev.split(':'):
      env['PATH'] = prev + ':' + path
  
  else:
    env = None # Current environment variables 
  
  if shell:
    cmd_args = ' '.join(cmd_args)
    
  if stdin and isinstance(stdin, str):
    stdin = open(stdin)

  if stdout and isinstance(stdout, str):
    stdout = open(stdout, 'w')

  if stderr and isinstance(stderr, str):
    stderr = open(stderr, 'a')
  
  if stderr is None and LOGGING:
    logging()
    stderr = LOG_FILE_OBJ
  
  if wait:
    subprocess.call(cmd_args, stdin=stdin, stdout=stdout, stderr=stderr, env=env, shell=shell)
      
  else:
    subprocess.Popen(cmd_args, stdin=stdin, stdout=stdout, stderr=stderr, env=env, shell=shell)


# #  Strings  # #

 
def get_rand_string(size):
  
  return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for i in range(size))


# #  Kmeans  # #

def kMeansSpread(data, k, thresh=1e-10, verbose=False):
  """k-means with better spread starting values"""
    
  n = len(data)
  index = np.random.randint(0, n)
  indices = set([index])
  
  influence = np.zeros(n)
  while len(indices) < k:
    diff = data - data[index]
    sumSq = (diff * diff).sum(axis=1) + 1.0
    influence += 1.0 / sumSq
    index = influence.argmin()
    
    while index in indices:
      index = np.random.randint(0, n)
    
    indices.add(index)    
  
  centers = np.vstack([data[i] for i in indices])
    
  return kMeans(data, k, centers, thresh, verbose)


def kMeans(data, k, centers=None, thresh=1e-10, verbose=False):
  """k-means clustering"""
    
  if centers is None:
    centers = np.array( np.random.choice(list(data), k, replace=False) )  # list() not needed in Python 2

  labels = np.empty(len(data), float)
  change = 1.0
  prev = []

  j = 0
  while change > thresh:

    clusters = [[] for x in range(k)]
    for i, vector in enumerate(data):
      diffs = centers - vector
      dists = (diffs * diffs).sum(axis=1)
      closest = dists.argmin()
      labels[i] = closest
      clusters[closest].append(vector)
     
    change = 0
    for i, cluster in enumerate(clusters):
      cluster = np.array(cluster)
      n = max(len(cluster), 1) # protect against being 0
      center = cluster.sum(axis=0)/n
      diff = center - centers[i]
      change += (diff * diff).sum()
      centers[i] = center
    
    j += 1
    
    if verbose:
      print(j, change) 
    
  return centers, clusters, labels


