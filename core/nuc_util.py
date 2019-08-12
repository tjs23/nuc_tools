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

# # Matplotlib # # 

COLORMAP_URL = 'https://matplotlib.org/tutorials/colors/colormaps.html'

def string_to_colormap(color_spec, cmap_name='user', n=255):
  
  from matplotlib.colors import LinearSegmentedColormap
  
  if isinstance(color_spec, (list,tuple)):
    colors = color_spec.split(',')
    try:
      cmap = LinearSegmentedColormap.from_list(name=cmap_name, colors=colors, N=n)    
    
    except ValueError as err:
      util.warn(err)
      util.critical('Invalid colour specification')
  
  elif ',' in color_spec:
    colors = color_spec.split(',')
    try:
      cmap = LinearSegmentedColormap.from_list(name=cmap_name, colors=colors, N=n)    
    
    except ValueError as err:
      util.warn(err)
      util.critical('Invalid colour specification')
    
  else:
    try:
      cmap = plt.get_cmap(cmap)
      
    except ValueError as err:
      util.warn(err)
      util.critical('Invalid colourmap name. See: %s' % COLORMAP_URL)
  
  return cmap
  

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


def downsample_matrix(in_array, new_shape, as_mean=False):
    
  p, q = in_array.shape
  n, m = new_shape
  
  if (p,q) == (n,m):
    return in_array
  
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
  
  if as_mean:
    return in_array.reshape(shape).mean(-1).mean(1)
  else:
    return in_array.reshape(shape).sum(-1).sum(1)


def sort_chromosomes(chromos):
  
  sort_chromos = []
  
  for chromo in chromos:
    if chromo.lower().startswith('chr'):
      c = chromo[3:]
    elif chromo.lower().startswith('scaffold'):
      c = chromo[8:]
    else:
      c = chromo

    if ('.' in c) and c.split('.')[-1].upper() in ('A','B'):
      try:
        key = ('%09d' % int(c.split('.')[0]), c.split('.')[-1])
      except ValueError as err:
        key = (c, c.split('.')[-1],)

    else:
      try:
        key = ('%09d' % int(c), 'a')
      except ValueError as err:
        key = (c, 'a')

    sort_chromos.append((key, chromo))
  
  sort_chromos.sort()
  
  return [x[1] for x in sort_chromos]  


def points_region_intersect(pos, regions):
  
  starts = np.array(regions[:,0])
  ends = np.array(regions[:,1])
  
  idx = np.argsort(starts)

  starts = starts[idx]
  ends = ends[idx]
  
  before_end = np.searchsorted(ends, pos)
  after_start = np.searchsorted(starts, pos)-1
  
  intersection = after_start == before_end
  
  return intersection
  
  
def bin_region_values(regions, values, bin_size, start, end):
  """
  Bin input regions and asscociated values into a histogram of new, regular
  regions. Accounts for partial overlap using proportinal allocation.
  """  
  n = len(values)
  
  if len(regions) != n:
    data = (len(regions), n)
    msg = 'Number of regions (%d) does not match number of values (%d)'
    raise Exception(msg % data)  
    
  s = int(start/bin_size)
  e = 1 + int(end/bin_size) # Limit, not last index
  n_bins = e-s
  value_hist = np.zeros(n_bins, float)
  
  if not len(regions):
    return value_hist
  
  np.sort(regions)
  sort_idx = regions[:,0].argsort()
  regions = regions[sort_idx]
  values = values[sort_idx]
  
  s *= bin_size
  e *= bin_size   
  boundaries = np.linspace(s,e,n_bins+1)
 
  starts = regions[:,0]
  ends = regions[:,1]  
  start_bin  = np.searchsorted(boundaries, starts)
  end_bin    = np.searchsorted(boundaries, ends)
  
  keep = (start_bin > 0) & (end_bin < n_bins) # Data often exceeds common structure regions

  mask = (end_bin == start_bin) & keep
  value_hist[start_bin[mask]] += values[mask]
  
  spanning = (~mask & keep).nonzero()[0]
  
  if len(spanning): # Overlapping cases (should be rare)
    for i in spanning:
      v = values[i]
      p1 = starts[i]
      p2 = ends[i]
      r = float(p2-p1)
 
      for j in range(start_bin[i], end_bin[i]+1): # Region ovelaps bins 
        p3 = s + j * bin_size # Bin start pos
        p4 = p3 + bin_size    # Bin limit
 
        if (p1 >= p3) and (p1 < p4): # Start of region in bin
          f = float(p4 - p1) / r
 
        elif (p2 >= p3) and (p2 < p4): # End of region in bin
          f = float(p2 - p3) / r
 
        elif (p1 < p3) and (p2 > p4): # Mid region in bin
          f = bin_size / r
 
        else:
          f = 0.0
 
        value_hist[j] += v * f
  
  return value_hist


def unpack_chromo_coords(coords, chromosomes, seq_pos_dict):
  """
  Exctract coords for multiple chromosomes stored in a single array into
  a dictionary, keyed by chromosome name. The chromosomes argument is required
  to get the correct array storage order.
  """

  chromo_num_particles = [len(seq_pos_dict[chromo]) for chromo in chromosomes]
  n_seq_pos = sum(chromo_num_particles)
  n_models, n_particles, dims = coords.shape

  if n_seq_pos != n_particles:
    msg = 'Model coordinates must be an array of num models x %d' % (n_seq_pos,)
    raise(Exception(msg))  
  
  coords_dict = {}
        
  j = 0
  for i, chromo in enumerate(chromosomes):
    span = chromo_num_particles[i]
    coords_dict[chromo] = coords[:,j:j+span] # all models, slice
    j += span
 
  return coords_dict


def pack_chromo_coords(coords_dict, chromosomes):
  """
  Place chromosome 3D coordinates stored in a dictionary keyed by
  chromosome name into a single, ordered array. The chromosomes argument
  is required to set the correct array storage order.
  """

  chromo_num_particles = [len(coords_dict[chromo][0]) for chromo in chromosomes]
  n_particles = sum(chromo_num_particles)
  n_models = len(coords_dict[chromosomes[0]])  
  coords = np.empty((n_models, n_particles, 3), float)
  
  j = 0
  for i, chromo in enumerate(chromosomes):
    span = chromo_num_particles[i]
    coords[:,j:j+span] = coords_dict[chromo]
    j += span
      
  return coords
  
  
def svd_rotate(coords_a, coords_b, weights=None):
  """
  Aligns coords_b to coords_a by rotation, returning transformed coords
  """
  
  coords_bt = coords_b.transpose()
  
  if weights is None:
    coords_bt = coords_b.transpose()
  else:
    coords_bt = coords_b.transpose() * weights

  mat = np.dot(coords_bt, coords_a)
  rot_mat1, _scales, rot_mat2 = np.linalg.svd(mat)
  sign = np.linalg.det(rot_mat1) * np.linalg.det(rot_mat2)

  if sign < 0:
    rot_mat1[:,2] *= -1
  
  rotation = np.dot(rot_mat1, rot_mat2)
  
  return np.dot(coords_b, rotation)
 
 
def center_coords(coords, weights):
  """
  Transpose coords to zero at centroid
  """
  
  wt_coords = coords.transpose() * weights
  xyz_totals = wt_coords.sum(axis=1)
  center = xyz_totals/sum(weights)
  cen_coords = coords - center
  
  return cen_coords


def calc_rmsds(ref_coords, coord_models, weights=None):
  """
  Calculates per model and per particle RMSDs compared to reference coords
  """
  
  n_coords = len(ref_coords)
  n_models = len(coord_models)
  
  if weights is None:
    weights = np.ones(n_coords)
    sum_weights = n_coords
  else:
    sum_weights = sum(weights)
  
  model_rmsds = []
  sum_deltas2 = np.zeros((n_coords, 3))
  
  for coords in coord_models:
    deltas2 = (coords-ref_coords)**2
    sum_deltas2 += deltas2
    dists2 = weights*deltas2.sum(axis=1)
    model_rmsds.append(np.sqrt(sum(dists2)/sum_weights))
  
  particle_rmsds = np.sqrt(sum_deltas2.sum(axis=1)/n_models)

  return np.array(model_rmsds), particle_rmsds
  
 
def align_coord_pair(coords_a, coords_b, dist_scale=1.0):
  """
  Align two coord arrays.
  Returns the transformed version of coords_a and coords_b.
  Returns the model and particle RMSD.
  """
  
  n = len(coords_a)
  weights = np.ones(n, float) # All particles intially weighted equally
  
  # Move both coords arrays to origin, original inputs are preserved
  coords_a = center_coords(coords_a, weights)
  coords_b = center_coords(coords_b, weights)
  
  # Align B to A and get RMDs of transformed coords
  coords_b1 = svd_rotate(coords_a, coords_b, weights)
  rmsd_1, particle_rmsds_1 = calc_rmsds(coords_a, [coords_b1], weights)

  # Align mirror B to A and get RMDs of transformed coords
  coords_b2 = svd_rotate(coords_a, -coords_b, weights)
  rmsd_2, particle_rmsds_2 = calc_rmsds(coords_a, [coords_b2], weights)

  if rmsd_1[0] < rmsd_2[0]:
    coords_b = coords_b1
    particle_rmsds = particle_rmsds_1
    
  else: # Mirror is best
    coords_b = coords_b2
    particle_rmsds = particle_rmsds_2
        
  # Refine alignment with exponential weights that deminish as RMSD increases
  if dist_scale:
    med_rmsd = np.median(particle_rmsds) # Gives a degree of scale invariance
    weight_scale = particle_rmsds / dist_scale
    weights_exp = np.exp(-weight_scale*weight_scale*med_rmsd)

    coords_a = center_coords(coords_a, weights_exp)
    coords_b = center_coords(coords_b, weights_exp)
    coords_b = svd_rotate(coords_a, coords_b, weights_exp)
    
  return coords_a, coords_b
  
  
def align_coord_models(coord_models, n_iter=1, dist_scale=True):
  """
  Aligns multiple coords arrays.
  Convergence is normally good with n_iter=1, but larger values may be needed in troublesome cases.
  Set the dist_scale to down-weight RMSD values that approach/exceed this size.
  The default dist_scale(=True) automatically sets a value to only ignore the worst outliers.
  Setting dist_scale to None or any false value disables all RMSD based weighting; this can give a better overall RMSD,
  at it tries to fit outliers, but worse alignment for the most invariant particles.
  Returns aligned coords models, RMSD of each model and RMSD of each particle relative to mean.
  """

  coord_models = np.array(coord_models)
  n_models, n_coords = coord_models.shape[:2]
      
  if dist_scale is True:
    init_dist_scale = 0.0
  else:
    init_dist_scale = dist_scale
  
  # Align to first model arbitrarily
  ref_coords = coord_models[0]
  for i, coords in enumerate(coord_models[1:], 1):
    coords_a, coords_b = align_coord_pair(ref_coords, coords, init_dist_scale)
    coord_models[i] = coords_b
    
  coord_models[0] = coords_a # First model has been centred

  # Align all coord models to closest to mean (i.e. a real model)
  # given initial mean could be wonky if first model was poor
  model_rmsds, particle_rmsds = calc_rmsds(coord_models.mean(axis=0), coord_models)
  
  # When automated the distance scale is set according to a large RMSD vale
  if dist_scale is True:
    dist_scale = np.percentile(particle_rmsds, [99.0])[0]
    
  j = np.array(model_rmsds).argmin()
  ref_coords = coord_models[j]
  
  for i, coords in enumerate(coord_models):
    if i != j:
      coords_a, coords_b = align_coord_pair(ref_coords, coords, dist_scale)
      coord_models[i] = coords_b
  
  # Align all coord models to mean and converge iteratively
  for j in range(n_iter):
    ref_coords = coord_models.mean(axis=0)
    for i, coords in enumerate(coord_models):
      coords_a, coords_b = align_coord_pair(ref_coords, coords, dist_scale)
      coord_models[i] = coords_b
      
  # Final mean for final RMSDs

  model_mean_rmsds, particle_rmsds = calc_rmsds(coord_models.mean(axis=0), coord_models)
  
  model_rmsds = np.zeros((n_models, n_models))

  for i in range(n_models-1):
    for j in range(i+1, n_models):
       rmsds, null = calc_rmsds(coord_models[i], [coord_models[j]])
       model_rmsds[i,j] = rmsds[0]
       model_rmsds[j,i] = rmsds[0]
  
  return coord_models, model_rmsds, model_mean_rmsds, particle_rmsds


def align_chromo_coords(coords_dict, seq_pos_dict, n_iter=1, dist_scale=True):
  
  chromos = sorted(seq_pos_dict)
  
  coord_models = pack_chromo_coords(coords_dict, chromos)
  
  coord_models, model_rmsds, model_mean_rmsds, particle_rmsds = align_coord_models(coord_models, n_iter, dist_scale)  
    
  coords_dict = unpack_chromo_coords(coord_models, chromos, seq_pos_dict)
  
  return coords_dict, model_rmsds, model_mean_rmsds, particle_rmsds
 



