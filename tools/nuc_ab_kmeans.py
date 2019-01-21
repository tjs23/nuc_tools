import sys, os
import numpy as np


from formats import bed, ncc, n3d
from core import nuc_util

PROG_NAME = 'nuc_ab_means'
VERSION = '1.0.0'
DESCRIPTION = 'Calculate the A/B regions from population Hi-C data'

DEFAULT_BIN_SIZE = 500000

def test_imports():
    
  try:
    import tools.cyt_ab_kmeans
    
  except ImportError as err:
    try:
      import cython
    except ImportError as err:
      nuc_util.critical('Critical Python module "cython" is not installed or accessible')

    try:
      from distutils.core import run_setup
    except ImportError as err:
      nuc_util.critical('Critical Python module "distutils" is not installed or accessible')
  
    from distutils.core import run_setup
    nuc_util.warn('Utility C/Cython code not compiled. Attempting to compile now...')    
    
    setup_script = os.path.join(os.path.dirname(__file__), 'setup_cython.py')
    run_setup(setup_script, ['build_ext', '--inplace'])  

  import cyt_ab_kmeans
  #import cyt_ncc


def calcContactVoidRegions(pop_contacts_path, binSize=int(5e5), close_cis=int(1e6), clip_factor=2.0):
  
  chromosomes, chromo_limits, contact_dict = ncc.load_file(pop_contacts_path, pair_key=False)

  regionDict = {}
  
  for chrA in contact_dict:
    if chrA not in contact_dict[chrA]: # Must have cis data
      continue
    
    limits = chromo_limits[chrA]
    start, end = limits

    num_bins = int((end-start)/binSize)
    if not num_bins: # A null entry  - can happen with MT and Y
      continue
    
    # Scaled, observed contact matrix, without diagonal
    obs = ncc.getContactMatrix(contact_dict, chrA, chrA, limits, limits, binSize).astype(float)
    obs -= np.diag(np.diag(obs))
    
    obs_sum = obs.sum()
    if not obs_sum: # Empty matrix
      continue
    
    obs /= obs_sum
    n = len(obs)
    
    # Bin contact data over full extent
    data = contact_dict[chrA][chrA]
    hist, edges = np.histogram(data[:2].ravel(), bins=num_bins, range=(start, end+binSize))            
    
    if len(hist) < 2:
      continue
    
    # Selection mask for which bin indices are void
    selection = np.zeros(len(hist), int)
    
    # Set void selection mask based on comparing binned contact data to median
    med = np.median(hist)

    idx = (hist < (1/clip_factor) * med).nonzero()
    selection[idx] = 1
    
    idx = (hist > (clip_factor) * med).nonzero()
    selection[idx] = 1
    
    # Select close in sequence for extra check - these regions should be strong
    seq_seps = abs(data[0]-data[1])
    
    idx = (seq_seps < close_cis).nonzero()[0]
    close_cis_data = data[:,idx]
    
    # Bin close contacts and set void mask where too sparse
    values = close_cis_data[:2].ravel() # All contact ends, i.e. flatten to diagonal
    c_hist, edges = np.histogram(values, bins=num_bins, range=(start, end+binSize))
    
    med = np.median(c_hist)
    idx = (c_hist < (1/clip_factor) * med).nonzero()
    selection[idx] = 1

    # void indices of contact matrix
    idx = selection.nonzero()

    # void region ranges
    starts = edges[:-1]
    ends = edges[1:]-1
    
    starts = starts[idx]
    ends = ends[idx]
    
    if len(starts) < 1: # No void in this chromosome
      continue
    
    # Dict of void regions for this chromosome
    regionDict[chrA] = np.array(list(zip(starts, ends)), np.int32)  # list() needed for Python 3
    
  return regionDict

def make_ab_compartment_tracks(pop_contacts_path, active_marks_track=None, binSize=int(2.5e5)):
  """Calculate A/B compartment regions from population Hi-C contacts"""
      
  import tools.cyt_ab_kmeans as cyt_ab_kmeans
  
  chromosomes, chromo_limits, contact_dict = ncc.load_file(pop_contacts_path, pair_key=False)
  if active_marks_track is None:
    marks_regions = marks_values = None
  else:
    marks_regions, marks_values, label_dict = bed.load_bed_data_track(active_marks_track)
    
  regionDictA = {}
  regionDictB = {}
  
  void_track_regions = calcContactVoidRegions(pop_contacts_path, binSize*4)

  for chromo in chromosomes:
    if chromo not in marks_regions:
      continue
    
    if chromo not in void_track_regions:
      continue
      
    print('working on chromo %s' % chromo)
    
    limits = chromo_limits[chromo]
    startPoint, endPoint = limits
    
    # Get observed contacts matrix, remove digonal and scale
    obs = ncc.getContactMatrix(contact_dict, chromo, chromo, limits, limits, binSize).astype(float)
    obs -= np.diag(np.diag(obs)) # Repeating diag() makes 1D into 2D
    obs /= obs.sum()
    n = len(obs)
    
    # Get total contacts signal and bin counts for each seq separation 
    counts = np.zeros(n, dtype='int')
    sig = np.zeros(n)
    
    for i in range(n):
      for j in range(i,n):
        d = j-i
        sig[d] += obs[i,j]
        counts[d] += 1
    
    # Get mean contacts signal at each seq sep and normalise to sum 1
    for c, j in enumerate(counts):
      if c:
        sig[j] /= c
    
    sig /= sig.sum()  
    
    # Initial expectation for mean normalised contacts genen the seq. separation
    #exp = np.arange((n,n), float) # does not work in Python 2.7 or Python 3.6
    exp = np.arange(n*n, dtype=np.float)
    exp = exp.reshape((n, n))
    for i in range(n):
      exp[i,:i+1] = sig[:i+1][::-1]
      exp[i,i:] = sig[:n-i]
    
    # Sum observed rows/cols to get 1D sequence 'sensitivity' values
    vals = obs.sum(axis=0)
    vals /= vals.sum()
    
    # Expectation multiplied prodict of sequence sensitivity values
    exp *= np.outer(vals, vals)
    exp /= exp.sum()  
   
    # Get seq regions positions for contact map bins
    pos = startPoint + np.arange(0, n+1) * binSize
    pos = np.vstack([pos[:-1],  pos[1:]-1]).T.astype(np.int32)
    
    # Selection mask for non-void regions
    idx_nv = np.ones(len(pos))
    
    # Zero the void regions of the contact map
    # shifts seq regions about a little to exclude edge cases
    void_regions = np.array(void_track_regions[chromo], np.int32)
    
    idx_v = cyt_ab_kmeans.pairRegionsIntersection(pos, void_regions, exclude=False, allow_partial=True)
    idx_nv[idx_v] = 0
    
    # Select only non-void indices
    idx_nv = idx_nv.nonzero()[0]
    obs = obs[idx_nv]
    obs = obs[:,idx_nv]
    exp = exp[idx_nv]
    exp = exp[:,idx_nv]
    
    # Only non-zero values can be used in log ratio
    idx = (exp * obs).nonzero()
    z = ((exp * obs) == 0.0).nonzero()
    
    # Log(obs/exp)
    log_ratio = obs
    log_ratio[idx] /= exp[idx]
    log_ratio[idx] = np.log(log_ratio[idx])
    log_ratio = np.clip(log_ratio, -4.0, 4.0)
        
    # Covarience matrix
    cov_mat = np.cov(log_ratio)
    cov_mat[z] = 0.0 
    
    # Cluster covarience into two groups
    centers, clusters, labels = nuc_util.kMeansSpread(cov_mat, 2, verbose=False)
    labels += 1
    
    # Bin active marker data into contact regions
    regions = np.array(marks_regions[chromo], np.int32) # start, end
    values = np.array(marks_values[chromo], np.float)
    binned_markers = cyt_ab_kmeans.regionBinValues(regions, values, np.int32(binSize), startPoint, endPoint)
    
    # Select non-void seq positions and marker data
    pos = pos[idx_nv]
    binned_markers = binned_markers[idx_nv]
    
    # Find group with largest overlap to active marker data
    idx_a = (labels == 1).nonzero()[0]
    idx_b = (labels == 2).nonzero()[0]
    
    if len(idx_a) > 0 and len(idx_b) > 0: # protect against being 0
      f_a = binned_markers[idx_a].sum()/len(idx_a)
      f_b = binned_markers[idx_b].sum()/len(idx_b)
  
      # Swap where needed so "A" compartment overlaps with active markers most
      if f_b > f_a:
        f_a, f_b = f_b, f_a
        idx_a, idx_b = idx_b, idx_a
    
    elif len(idx_b) > 0:
      idx_a, idx_b = idx_b, idx_a
      
    pos_a = pos[idx_a]
    pos_b = pos[idx_b]
    
    # Fill A/B compartment output dictionaries
    regionDictA[chromo] = pos_a
    regionDictB[chromo] = pos_b
  
  return regionDictA, regionDictB, void_track_regions, chromo_limits

def save_ab_tracks_bed(region_dict_a, region_dict_b, output_prefix):
  
  value_dict_a = {}  
  for chromo in region_dict_a:
    x_vals = region_dict_a[chromo][:,0]
    y_vals = np.ones(x_vals.shape)
    value_dict_a[chromo] = y_vals

  value_dict_b = {}
  for chromo in region_dict_b:
    x_vals = region_dict_b[chromo][:,0]
    y_vals = np.ones(x_vals.shape)
    value_dict_b[chromo] = y_vals

  file_name = '%s_A.bed' % output_prefix
  bed.save_bed_data_track(file_name, region_dict_a, value_dict_a)

  file_name = '%s_B.bed' % output_prefix
  bed.save_bed_data_track(file_name, region_dict_b, value_dict_b)

def main(argv=None):
  
  from argparse import ArgumentParser

  if argv is None:
    argv = sys.argv[1:]
  
  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'
  arg_parse = ArgumentParser(prog='nuc_tools ab_kmeans', description=DESCRIPTION,
                            epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument('-ncc_path', nargs=1, metavar='NCC_FILE',
                         help='Input NCC format file containing single-cell Hi-C contact data')

  arg_parse.add_argument('-marks_track', nargs=1, metavar='TRACK_FILE',
                         help='Input active marks track file')

  arg_parse.add_argument('-bin_size',  metavar='BIN_SIZE', default=DEFAULT_BIN_SIZE, type=int,
                         help='Optional bin size for A/B calculation. Default %s' % DEFAULT_BIN_SIZE)

  arg_parse.add_argument('-o',  metavar='OUTPUT_PREFIX',
                         help='Optional output file prefix. ' \
                              'Unless specified the output will be the input ncc path prefix')  
 
  args = vars(arg_parse.parse_args(argv))
  
  pop_contacts_path     = args['ncc_path'][0]
  active_marks_track    = args['marks_track'][0]
  bin_size              = args['bin_size']
  output_prefix         = args['o']
  
  if not output_prefix:
    n = pop_contacts_path.rfind('.')
    if n > 0:
      output_prefix = pop_contacts_path[:n]
    else:
      output_prefix = pop_contacts_path

  a_regions, b_regions, v_regions, chromo_limits = make_ab_compartment_tracks(pop_contacts_path, active_marks_track, bin_size)

  save_ab_tracks_bed(a_regions, b_regions, output_prefix)

test_imports()
  
if __name__ == '__main__':
    
  main()


