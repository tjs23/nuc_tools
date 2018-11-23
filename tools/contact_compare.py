import os, sys, math
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict 
from scipy import sparse

PROG_NAME = 'contact_compare'
VERSION = '1.0.0'
DESCRIPTION = 'Compare two Hi-C contact maps (NPZ format)'

def normalize_contacts(contact_dict, chromo_limits, bin_sizes, bin_size, cis_only=False, clip=0.4):
  """
  For now dict is changed in-place to keep memory use down
  """
  from nuc_tools import util, io
  
  contact_scale = {}
  chromo_offsets = {}
  chromos = sorted(bin_sizes) # contact dict pair keys will always be in alphabetic order
  
  for chr_a in chromos:
    s, e = chromo_limits[chr_a]
    off = int(s/bin_size)
    chromo_offsets[chr_a] = off
    contact_scale[chr_a] = np.zeros(bin_sizes[chr_a], float) # Always start from zero
  
  # Get row sums over whole map
  
  util.info(' .. fetch scalings', line_return=True)
  pairs = []
  
  for k, chr_a in enumerate(chromos):
    for chr_b in chromos[k:]:
      pair = (chr_a, chr_b)
      orig_mat = contact_dict.get(pair)
      
      if orig_mat is None:
        continue
      
      a, b = orig_mat.shape
      pairs.append(pair)
      off_a = chromo_offsets[chr_a]
      off_b = chromo_offsets[chr_b]
      
      contact_scale[chr_a][off_a:off_a+a] += orig_mat.sum(axis=1)
      contact_scale[chr_b][off_b:off_b+b] += orig_mat.sum(axis=0)
  
  # Make reciprocal and remove void regions
        
  for chr_a in contact_scale:
    scale = contact_scale[chr_a]
    med = np.median(scale)
    
    too_small = scale < (clip*med)
    too_large = scale > (med/clip)
    
    scale[scale == 0] = 1.0
    scale = 1.0/scale
 
    scale[too_small] = 0.0
    scale[too_large] = 0.0
    
    contact_scale[chr_a] = scale
    
  
  for chr_a, chr_b in pairs: # Sorted and avliable
    is_cis = chr_a == chr_b
    
    if cis_only and not is_cis:
      del contact_dict[(chr_a, chr_b)]
      continue
    
    util.info(' .. {} {}   '.format(chr_a, chr_b), line_return=True)
    mat = contact_dict[(chr_a, chr_b)].astype(np.float32)
    a, b = mat.shape
    off_a = chromo_offsets[chr_a]
    lim_a = bin_sizes[chr_a]
    off_b = chromo_offsets[chr_b]
    lim_b = bin_sizes[chr_b]
    
    if off_a or off_b or (lim_a-a-off_a) or (lim_b-b-off_b):
      # all pairs use full range from zero
      mat = np.pad(mat, [(off_a,lim_a-a-off_a), (off_b,lim_b-b-off_b)], 'constant') # will ensure square cis (it needn't be when only storing upper matrix)
      a, b = mat.shape

    if is_cis:
      mat -= np.diag(np.diag(mat))
      
      for i in range(1,a):
        if mat[i,i-1]: # Check data is present below the diagonal
          contact_scale[chr_a] *= 2 # Everything was counted twice
          break
      
      else:
        mat += mat.T
        
    scale_a = contact_scale[chr_a]
    scale_b = contact_scale[chr_b]
    
    mat *= np.sqrt(np.outer(scale_a, scale_b))

    nnz = np.sqrt(len(scale_a.nonzero()[0]) * len(scale_b.nonzero()[0]))
    
    mat *= nnz/(mat.sum() or 1.0) # Counts scale with chromosome sizes
    
    if is_cis:
      mat = sparse.csr_matrix(mat)
    else:
      mat = sparse.coo_matrix(mat)
    
    contact_dict[(chr_a, chr_b)] = mat
    

def _vanilla_norm_cis(orig_mat, start_bin, n, clip=0.4):
  
  a, b = orig_mat.shape
  
  mat = np.zeros((n, n), float)
  mat[start_bin:a+start_bin,start_bin:b+start_bin] += orig_mat
  mat += mat.T # Only one side of diagonal was stored
  mat -= np.diag(np.diag(mat)) # Repeating diag() makes 1D into 2D

  scale = mat.sum(axis=0)
  
  med = np.median(scale)
  
  too_small = scale < (clip * med)
  too_large = scale > (med/clip)
  scale[scale == 0] = 1.0
  scale = 1.0/scale 
  
  scale[too_small] = 0.0
  scale[too_large] = 0.0
  
  mat *= np.sqrt(np.outer(scale, scale))

  nnz = len(scale.nonzero()[0])
  mat *= float(nnz)/(mat.sum() or 1.0) # Counts scale with chromosome size

  return mat

def _get_expectation(sig_seps, n):
    
  expt = np.zeros((n, n), float)
  for i in range(n):
    expt[i,:i] = sig_seps[:i][::-1]
    expt[i,i:] = sig_seps[:n-i]
    
  return expt  
  
  
def contact_compare(in_path_a, in_path_b, out_path=None):

  from nuc_tools import util, io
  from formats import npz  
  
  if not out_path:
    out_path = io.merge_file_names(in_path_a, in_path_b)
    
  file_bin_size_a, chromo_limits_a, contacts_a = npz.load_npz_contacts(in_path_a)
  file_bin_size_b, chromo_limits_b, contacts_b = npz.load_npz_contacts(in_path_b)

  if file_bin_size_a != file_bin_size_b:
    util.critical('Chromatin contact matrices to be compared must be beinned at the same resolution')
  
  chromos = sorted(set(chromo_limits_a.keys()) & set(chromo_limits_b.keys()))
  chromos = [c for c in chromos if (c,c) in contacts_a and (c,c) in contacts_b]
  
  if not chromos:
    util.critical('No chromosome names are common to both datasets')
  
  out_matrix = {}
  chromo_limits = {}
  
  cis_pairs = []
  trans_pairs = []
  
  for k, chr_a in enumerate(chromos):
    for chr_b in chromos[k:]:
      key = (chr_a, chr_b)
      
      if (key in contacts_a) and (key in contacts_b):
        if chr_a == chr_b:
          cis_pairs.append(key)
        else:
          trans_pairs.append(key)  
  
  util.info('Normalisation')
  
  sizes = {}
  for key in cis_pairs:
    s1, e1 = chromo_limits_a[key[0]]
    s2, e2 = chromo_limits_b[key[0]]
    chromo_limits[key[0]] = (0, max(e1, e2))
    sizes[key[0]] = int(math.ceil(max(e1, e2)/file_bin_size_a))
  
  normalize_contacts(contacts_a, chromo_limits_a, sizes, file_bin_size_a, cis_only=False)
  normalize_contacts(contacts_b, chromo_limits_b, sizes, file_bin_size_b, cis_only=False)  
        
  util.info('Calulating differences')
  
  for key in cis_pairs:
    util.info(' .. {}'.format(key[0]), line_return=True)  
       
    obs_a = contacts_a[key].toarray()
    obs_b = contacts_b[key].toarray()
    
    n, m = obs_a.shape
    
    """
    sep_dict_a = defaultdict(list)
    sep_dict_b = defaultdict(list)
    #sep_sig_a = np.zeros(n, float)
    #sep_sig_b = np.zeros(n, float)
    
    for d in range(1, n):
      idx1 = np.array(range(n-d))
      idx2 = idx1 + d
      idx = (idx1, idx2)
      sep_dict_a[d] = obs_a[idx]
      sep_dict_b[d] = obs_b[idx]
    """
    obs_a[obs_a < 1.0/n] = 0.0 # Avoid anything too small to avoid extremes
    obs_b[obs_b < 1.0/n] = 0.0          
    diff = np.zeros((n, n), np.float32)

    for d in range(1, n):
      z_scores = np.zeros(n-d, np.float32)
      idx1 = np.array(range(n-d))
      idx2 = idx1 + d
      idx = (idx1, idx2)

      vals_a = obs_a[idx]
      vals_b = obs_b[idx]
      
      nz = (vals_a * vals_b).nonzero()
            
      if len(nz[0]) > 2:
        nz_deltas = vals_a[nz] - vals_b[nz]
        med = np.median(nz_deltas)
        std = 1.4826 * np.median(np.abs(nz_deltas-med))
        z = (nz_deltas - med)/(std or 1.0)
        z[z > 5.0] = 5.0
        z[z < -5.0] = -5.0
        z_scores[nz] = z
               
      diff[idx] = z_scores
     
    """
    sep_sig_a = np.zeros(n, float)
    sep_sig_b = np.zeros(n, float)
    for i in range(n):
      if i in sep_dict_a:
        sep_sig_a[i] = np.median(sep_dict_a[i]) # already non-zero
    
      if i in sep_dict_b:
        sep_sig_b[i] = np.median(sep_dict_b[i])

    exp_a = _get_expectation(sep_sig_a, n)
    exp_b = _get_expectation(sep_sig_b, n)
    
    nz_a = (exp_a * obs_a).nonzero()
    nz_b = (exp_b * obs_b).nonzero()

    obs_a[nz_a] /= exp_a[nz_a]
    obs_b[nz_b] /= exp_b[nz_b]
    
    nz = (obs_a * obs_b).nonzero()

    diff = np.zeros((n, n), float)
    diff[nz] = 0.5 * (obs_a[nz] + obs_b[nz]) *  np.log(obs_a[nz]/obs_b[nz])
    #diff[nz] = np.log(obs_a[nz]/obs_b[nz])
    """
    
    out_matrix[key] = sparse.csr_matrix(diff)
  
  util.info(' .. done {} chromosomes'.format(len(cis_pairs)), line_return=True)  
 
  for key in trans_pairs:
    util.info(' .. {} - {}'.format(*key), line_return=True)  
       
    obs_a = contacts_a[key].toarray()
    obs_b = contacts_b[key].toarray()
    
    n, m = obs_a.shape
    
    obs_a[obs_a < 1.0/n] = 0.0 
    obs_b[obs_b < 1.0/n] = 0.0
          
    z_scores = np.zeros((n, m), np.float32)
    
    nz = (obs_a * obs_b).nonzero()
    
    if len(nz[0]) > 2:
      nz_deltas = obs_a[nz] - obs_b[nz]
      med = np.median(nz_deltas)
      std = 1.4826 * np.median(np.abs(nz_deltas-med))
      
      z_scores[nz] = (obs_a[nz] - obs_b[nz] - med)/(std or 1.0)
      z_scores[z_scores > 5.0] = 5.0
      z_scores[z_scores < -5.0] = -5.0
    
    out_matrix[key] = sparse.coo_matrix(z_scores)

  util.info(' .. done {} pairs'.format(len(trans_pairs)), line_return=True)  
 
  util.info('Saving data')
    
  npz.save_contacts(out_path, out_matrix, chromo_limits, file_bin_size_a, min_bins=0)  
  util.info('Written {}'.format(out_path))
  

# Look at replicates plot difference/variance vs mean for each seq sep
# - Test both normlised obs and expectation relative obs
# - Test differences for normality
# Sample A vs B differences compared with replica 


def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument(metavar='CONTACT_FILES', nargs=2, dest='i',
                         help='Two input NPZ format (binned, bulk Hi-C data) chromatin contact files to be compared.')

  arg_parse.add_argument('-o', metavar='OUT_FILE', default=None,
                         help='Optional output PNPZ format file name. If not specified, a default based on the input file names will be used.')

  args = vars(arg_parse.parse_args(argv))

  in_path_a, in_path_b = args['i']
  out_path = args['o']
  
  contact_compare(in_path_a, in_path_b, out_path)
  
  
  
if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()
