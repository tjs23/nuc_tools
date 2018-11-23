import random, sys, string, subprocess
import uuid
import numpy as np

# #   Globals  # #

QUIET            = False # Global verbosity flag
LOGGING          = False # Global file logging flag
TEMP_ID          = '%s' % uuid.uuid4()

LOG_FILE_PATH    = 'nuc-tools-out-%s.log' % TEMP_ID
LOG_FILE_OBJ     = None # Created when needed

import core.nuc_parallel as parallel

# #   Srcreen reporting  # # 

NEWLINE_CHARS = 0

def report(msg, line_return):
 
  global LOG_FILE_OBJ
  global NEWLINE_CHARS
  
  if LOGGING:
    if not LOG_FILE_OBJ:
      LOG_FILE_OBJ = open(LOG_FILE_PATH, 'w')
      
    LOG_FILE_OBJ.write(msg)
  
  if not QUIET:
    if line_return:
      fmt = '\r%%-%ds' % max(NEWLINE_CHARS, len(msg))
      sys.stdout.write(fmt % msg) # Must have enouch columns to cover previous msg
      sys.stdout.flush()
      NEWLINE_CHARS = len(msg)
    else: 
      if NEWLINE_CHARS:
        print('')
      print(msg)
      NEWLINE_CHARS = 0
   
def warn(msg, prefix='WARNING', line_return=False):

  report('%s: %s' % (prefix, msg), line_return)

 
def critical(msg, prefix='FAILURE', line_return=False):

  report('%s: %s' % (prefix, msg), line_return)
  sys.exit(0)


def info(msg, prefix='INFO', line_return=False):

  report('%s: %s' % (prefix, msg), line_return)


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


def downsample_matrix(in_array, new_shape):
    
    p, q = in_array.shape
    n, m = new_shape
    
    if p % n == 0:
      pad_a = 0
    else:
      pad_a = n * int(1+p//n) - p

    if q % m == 0:
      pad_b = 0
    else:
      pad_b = m * int(1+q//m) - q 
    
    if pad_a or pad_b:
      in_array = np.pad(in_array, [(0,pad_a), (0,pad_b)], 'constant')
      p, q = in_array.shape
        
    shape = (n, p // n,
             m, q // m)
    
    return in_array.reshape(shape).sum(-1).sum(1)
