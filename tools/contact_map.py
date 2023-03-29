import datetime
import numpy as np
import os, sys, re, math

from math import ceil, floor
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.linewidth'] = 0.5

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Colormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator

PROG_NAME = 'contact_map'
VERSION = '1.1.1'
DESCRIPTION = 'Chromatin contact (NCC or NPZ format) Hi-C contact map PDF display module'
DEFAULT_CIS_BIN_KB = 250
DEFAULT_TRANS_BIN_KB = 500
DEFAULT_MAIN_BIN_KB = 1000
DEFAULT_SC_MAIN_BIN_KB = 5000
DEFAULT_SC_CHR_BIN_KB = 250
DEFAULT_SMALLEST_CONTIG = 0.1
DEFAULT_DIAG_REGION = 50.0
COLORMAP_URL = 'https://matplotlib.org/tutorials/colors/colormaps.html'
REGION_PATT = re.compile('(\S+):(\d+\.?\d*)-(\d+\.?\d*)')
MIN_REGION_BINS = 10
DT_COLORMAP = '#000000,#0080FF,#B0B000,#FF4000,#FF00E0'
GFF_FORMAT    = 'GFF'
BED_FORMAT    = 'BED'
WIG_FORMAT     = 'Wiggle'
SAM_FORMAT    = 'BAM/SAM'
SAM_BIN_SIZE  = 1000

import warnings
warnings.filterwarnings("ignore")

def normalize_contacts(contact_dict, chromo_limits, bin_size, new_chromo_limits=None,
                       new_bin_size=None, compare_trans=False, clip=0.1, store_sparse=True):
  """
  For now dict is changed in-place to keep memory use down.
  """
  from nuc_tools import util, io
  
  if not new_bin_size:
    new_bin_size = bin_size
  
  if not new_chromo_limits:
    new_chromo_limits = chromo_limits
  
  chromo_sizes = {}
  contact_scale = {}
  chromo_offsets = {}
  
  chromos = sorted(new_chromo_limits) # contact dict pair keys will always be in alphabetic order

  for chr_a in chromos:
    s, e = chromo_limits[chr_a]
    off = int(s/bin_size) # Offset in the original data
    chromo_offsets[chr_a] = off
    
    s, e = new_chromo_limits[chr_a] # Range in new data
    num_bins = int(math.ceil(e/bin_size)) 
    contact_scale[chr_a] = np.zeros(num_bins, float) # Always start from zero
    chromo_sizes[chr_a] = num_bins
     
  # Get row sums over whole map
  
  util.info(' .. fetch scalings', line_return=True)
  pairs = []
  
  for k, chr_a in enumerate(chromos):
    for chr_b in chromos[k:]:
      pair = (chr_a, chr_b)
      orig_mat = contact_dict.get(pair)
      
      if orig_mat is None:
        continue
  
      if hasattr(orig_mat, 'toarray'):
        orig_mat = orig_mat.toarray()
          
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
    
    if (not compare_trans) and (not is_cis):
      del contact_dict[(chr_a, chr_b)]
      continue
    
    util.info(' .. {} {}   '.format(chr_a, chr_b), line_return=True)
    mat = contact_dict[(chr_a, chr_b)]
    
    if hasattr(mat, 'toarray'):
      mat = mat.toarray()
    
    mat = mat.astype(np.float32)
    a, b = mat.shape
    off_a = chromo_offsets[chr_a]
    lim_a = chromo_sizes[chr_a]
    off_b = chromo_offsets[chr_b]
    lim_b = chromo_sizes[chr_b]
    
    if off_a or off_b or (lim_a-a-off_a) or (lim_b-b-off_b):
      # all pairs use full range from zero
      mat = np.pad(mat, [(off_a,lim_a-a-off_a), (off_b,lim_b-b-off_b)], 'constant') # will ensure square cis (it needn't be when only storing upper matrix)
      a, b = mat.shape

    if is_cis:
      mat -= np.diag(np.diag(mat))
      
      for i in range(1,a):
        if mat[i,i-1]: # Check data is present below the diagonal
          contact_scale[chr_a] *= 2 # Everything was counted twice : divide by double the amount
          break
      
      else:
        mat += mat.T
        
    scale_a = contact_scale[chr_a].astype(np.float32)
    scale_b = contact_scale[chr_b].astype(np.float32)
    
    mat *= np.sqrt(np.outer(scale_a, scale_b))
    
    nnz = np.sqrt(len(scale_a.nonzero()[0]) * len(scale_b.nonzero()[0]))
    
    msum = mat.sum()
    
    if not msum:
      continue
    
    mat *= nnz/msum # The counts scale with the chromosome sizes
    
    if new_bin_size > bin_size: # i.e. do nothing if smaller or equal (smaller is not valid)
      ratio = bin_size / float(new_bin_size)
      p = int(math.ceil(a * ratio))
      q = int(math.ceil(b * ratio))
      mat = util.downsample_matrix(mat, (p, q))
       
    if store_sparse:
      if is_cis:
        mat = sparse.csr_matrix(mat)
      else:
        mat = sparse.coo_matrix(mat)
    
    contact_dict[(chr_a, chr_b)] = mat
  
  util.info(' .. normalised {} chromosomes/pairs'.format(len(pairs)), line_return=True)


def _downsample_matrix(in_array, new_shape, pad=False):
  
  from scipy.ndimage import zoom
    
  p, q = in_array.shape
  n, m = new_shape

  if (p,q) == (n,m):
    return in_array
  
  if (p % n == 0) and (q % m == 0):
    shape = (n, p // n, m, q // m)
    mat = in_array.reshape(shape).sum(-1).sum(1)
  
  elif pad:
    pad_a = 0 if p % n == 0 else n * int(1+p//n) - p
    pad_b = 0 if q % m == 0 else m * int(1+q//m) - q 
    in_array = np.pad(in_array, [(0,pad_a), (0,pad_b)], 'constant')
    p, q = in_array.shape
    shape = (n, p // n, m, q // m)
    mat = in_array.reshape(shape).sum(-1).sum(1)
  
  else:
    count = in_array.sum()
    mat = zoom(in_array, (n/float(p), m/float(q)), output=float, order=3,
               mode='constant', cval=0.0, prefilter=True)
    
    mat = np.clip(mat, 0.0, None)
    mat *= count/float(mat.sum())
    
  return mat


def _get_chromo_offsets(bin_size, chromos, chromo_limits):
  
  chromo_offsets = {}
  label_pos = []
  n = 0
  for chromo in chromos: # In display order
    s, e = chromo_limits[chromo]
    c_bins = int(ceil(e/float(bin_size))) - int(s/bin_size)
    chromo_offsets[chromo] = s, n, c_bins # Start bp, start bin index, num_bins
    label_pos.append(n + c_bins/2)
    n += c_bins
    #n += 1 # Add space between chromos on matrix
  
  return n, chromo_offsets, label_pos


def get_obs_vs_exp(obs, clip=10):

  obs -= np.diag(np.diag(obs))
  expt = get_cis_expectation(obs)

  prod = expt * obs

  nz = prod != 0.0
 
  log_ratio = obs.copy()
  log_ratio[nz] /= expt[nz]
  log_ratio[nz] = np.log(log_ratio[nz])
  log_ratio = np.clip(log_ratio, -clip, clip)
    
  return log_ratio
  
  
def get_trans_expectation(obs):
  
  vals0 = obs.sum(axis=0).astype(float)
  vals0 /= vals0.sum() or 1.0
 
  vals1 = obs.sum(axis=1).astype(float)
  vals1 /= vals1.sum() or 1.0

  expt = np.outer(vals1, vals0)
  expt *= obs.sum()/expt.sum()
  
  return expt
  
  
def get_cis_expectation(obs):

  n = len(obs)
  sobs = obs.sum()
  sep_dict = defaultdict(list) 
 
  for d in range(1, n):
    idx1 = np.array(range(n-d))
    idx2 = idx1 + d
    idx = (idx1, idx2)
    sep_dict[d] = obs[idx]
 
  sep_sig = np.zeros(n, float)
 
  for i in range(n):
    if i in sep_dict:
      sep_sig[i] = np.mean(sep_dict[i])

  expt = np.zeros((n, n), float)

  for i in range(n):
    expt[i,:i] = sep_sig[:i][::-1]
    expt[i,i:] = sep_sig[:n-i]
  
  vals = obs.sum(axis=0).astype(float)
  vals /= vals.sum() or 1.0
 
  expt *= np.outer(vals, vals)
  expt *= sobs/expt.sum()
  
  return expt

  
def get_corr_mat(obs, chromo_offsets=None, clip=5.0):

  obs -= np.diag(np.diag(obs))
  
  if chromo_offsets:
    expt = np.zeros(obs.shape)
    for chr_a in chromo_offsets:
      s1, i, a = chromo_offsets[chr_a] # First base, first bin, num bins
      
      for chr_b in chromo_offsets:
        s2, j, b = chromo_offsets[chr_b]
        
        if chr_a == chr_b:
          expt[i:i+a,i:i+a] = get_cis_expectation(obs[i:i+a,i:i+a])
        else:
          expt[i:i+a,j:j+b] = get_trans_expectation(obs[i:i+a,j:j+b])
      
  else: # All cis
    expt = get_cis_expectation(obs)

  prod = expt * obs
  nz = prod != 0.0
 
  log_ratio = obs.copy()
  log_ratio[nz] /= expt[nz]
  log_ratio[nz] = np.log(log_ratio[nz])
  log_ratio = np.clip(log_ratio, -clip, clip)
 
  corr_mat = np.corrcoef(log_ratio)
  corr_mat -= np.diag(np.diag(corr_mat))

  corr_mat = np.nan_to_num(corr_mat)
  
  return corr_mat


def get_trans_corr_mat(obs_a, obs_b, obs_ab, clip=5.0):
  
  n = len(obs_a)
  m = len(obs_b)
  z = n+m

  obs_a -= np.diag(np.diag(obs_a))
  obs_b -= np.diag(np.diag(obs_b))
  mat = np.zeros((z, z))
  
  # Add cis A
  
  expt = get_cis_expectation(obs_a)
  nz = (expt * obs_a) != 0.0  
  log_ratio = obs_a.copy()
  log_ratio[nz] /= expt[nz]
  
  mat[:n,:n] = log_ratio
  
  # Add cis B
  
  expt = get_cis_expectation(obs_b)
  nz = (expt * obs_b) != 0.0  
  log_ratio = obs_b.copy()
  log_ratio[nz] /= expt[nz]
  
  mat[n:,n:] = log_ratio
  
  # Add trans
  
  expt = np.full((n,m), obs_ab.mean())
  sobs = obs_ab.sum()
  
  vals_a = obs_a.sum(axis=0) + obs_ab.sum(axis=1)
  vals_b = obs_b.sum(axis=0) + obs_ab.sum(axis=0)
 
  expt *= np.outer(vals_a, vals_b)
  expt *= sobs/expt.sum()
  
  nz = (expt * obs_ab) != 0.0  
  log_ratio = obs_ab.copy()
  log_ratio[nz] /= expt[nz]
  
  mat[:n,n:] = log_ratio
  mat[n:,:n] = log_ratio.T
  
  nz = mat.nonzero()
  mat[nz] = np.log(mat[nz])
  mat = np.clip(mat, -clip, clip)
  
  corr_mat = np.corrcoef(mat)
    
  corr_mat = corr_mat[:n,n:]
    
  np.nan_to_num(corr_mat, copy=False)
  
  return corr_mat


def limit_counts(mat, limit):
  
  
  n, m = mat.shape
  
  if n == m:
    mat -= np.diag(np.diag(mat)//2)
    mat = np.triu(mat)
  
  count = mat.sum()
  nz = (mat > 0).nonzero()
  flat = mat[nz]
  rand_points = np.random.uniform(0.0, count-1.0, limit).astype(int)
  uniq, counts = np.unique(np.searchsorted(np.cumsum(flat), rand_points), return_counts=True)
    
  flat = np.zeros(flat.shape)
  flat[uniq] += counts
  mat[nz] = flat
  
  if n == m:
    mat += mat.T
    
  return mat
     
    
def _adjust_matrix_limits(contact_matrix, new_chromo_limits, orig_chromo_limits, orig_bin_size):

  s1a, e1a, s1b, e1b = new_chromo_limits # chr_a then chr_b
  s2a, e2a, s2b, e2b = orig_chromo_limits

  na = int(e1a/orig_bin_size) - int(s1a/orig_bin_size)
  nb = int(e1b/orig_bin_size) - int(s1b/orig_bin_size)
          
  s1 = max(0, int(s2a/orig_bin_size)-int(s1a/orig_bin_size))
  s2 = max(0, int(s2b/orig_bin_size)-int(s1b/orig_bin_size))
  s3 = max(0, int(s1a/orig_bin_size)-int(s2a/orig_bin_size))
  s4 = max(0, int(s1b/orig_bin_size)-int(s2b/orig_bin_size))
  
  da = min(na-s1, contact_matrix.shape[0]-s3)
  db = min(nb-s2, contact_matrix.shape[1]-s4)
  
  adjust_matrix = np.zeros((na, nb), float)
  adjust_matrix[s1:s1+da,s2:s2+db] = contact_matrix[s3:s3+da,s4:s4+db]
  
  return adjust_matrix


def get_contact_arrays_matrix(contacts, bin_size, chromos, chromo_limits,
                              orig_bin_size=None, orig_chromo_limits=None):
 
  n, chromo_offsets, label_pos = _get_chromo_offsets(bin_size, chromos, chromo_limits)
  
  matrix = np.zeros((n, n), float)
     
  n_ambig = 0
  n_homolog = 0
  n_trans = 0
  n_cis = 0
  n_cont = 0
  
  for i, chr_1 in enumerate(chromos):
    for chr_2 in chromos[i:]:

      if chr_1 > chr_2:
        chr_a, chr_b = chr_2, chr_1
      else:
        chr_a, chr_b = chr_1, chr_2

      contact_matrix = contacts.get((chr_a, chr_b))

      if contact_matrix is None:
        continue
     
      bp_a, bin_a, size_a = chromo_offsets[chr_a]
      bp_b, bin_b, size_b = chromo_offsets[chr_b]
      
      if orig_chromo_limits:
        lim_new = chromo_limits[chr_a]+chromo_limits[chr_b]
        lim_orig = orig_chromo_limits[chr_a]+orig_chromo_limits[chr_b]
        contact_matrix = _adjust_matrix_limits(contact_matrix, lim_new, lim_orig, orig_bin_size)
        
      count = contact_matrix.sum()
      
      if count == 0.0:
        continue
 
      sub_mat = _downsample_matrix(contact_matrix, (size_a, size_b), pad=True)
      
      matrix[bin_a:bin_a+size_a,bin_b:bin_b+size_b] += sub_mat
      matrix[bin_b:bin_b+size_b,bin_a:bin_a+size_a] += sub_mat.T
      
      count = int(count)
      
      if chr_a != chr_b:
        if ('.' in chr_a) and ('.' in chr_b) and (chr_a.split('.')[0] == chr_b.split('.')[0]):
          n_homolog += count

        n_trans += count

      else:
        n_cis += count
      
      n_cont += count 
  
  return (n_cont, n_cis, n_trans, n_homolog, n_ambig), matrix, label_pos, chromo_offsets


def _limits_to_shape(limits_a, limits_b, bin_size):
  
  start_a, end_a = limits_a
  start_b, end_b = limits_b
  
  n = int(ceil(end_a/float(bin_size))) - int(start_a/bin_size)
  m = int(ceil(end_b/float(bin_size))) - int(start_b/bin_size)
  
  return n, m


def get_single_array_matrix(contact_matrix, limits_a, limits_b, is_cis, orig_bin_size, bin_size, orig_chromo_limits=None):
  
  n, m = _limits_to_shape(limits_a, limits_b, bin_size)
  
  if orig_chromo_limits:
    contact_matrix = _adjust_matrix_limits(contact_matrix, limits_a+limits_b, orig_chromo_limits, orig_bin_size)
 
  if is_cis:
    n = m = max(n, m) # Square
  
  if bin_size == orig_bin_size:
    matrix = np.zeros((n, m), float)
    a, b = contact_matrix.shape
    matrix[:a,:b] += contact_matrix

  else:
    matrix = _downsample_matrix(contact_matrix.astype(float), (n, m), pad=True)
  
  return matrix
  

def get_single_list_matrix(contact_list, limits_a, limits_b, is_cis, bin_size, ambig_groups, smooth=False):
    
  start_a, end_a = limits_a
  start_b, end_b = limits_b
  
  hw = bin_size/2.0
  start_a -= hw
  start_b -= hw
  end_a += hw
  end_b += hw
  
  n, m = _limits_to_shape((start_a, end_a), (start_b, end_b), bin_size)
  matrix = np.zeros((n, m), float)
  ambig_matrix = np.zeros((n, m), float)
  
  if smooth:
    for p_a, p_b, nobs, ag in contact_list:
      p = (p_a-start_a)/bin_size
      q = (p_b-start_b)/bin_size
      
      a = int(p)
      b = int(q)
      
      f1 = p-a
      g1 = q-b
      
      if f1 > 0.5:
        c = a+1     # Alternative bin
        f1 = 1.5-f1 # Fraction in the main bin
        
      elif f1 < 0.5:
        c = a-1       # Alternative bin
        f1 = f1 + 0.5 # Fraction in the main bin
      else:
        c = a
      
      if g1 > 0.5:
        d = b+1
        g1 = 1.5-g1
        
      elif g1 < 0.5:
        d = b-1
        g1 = g1 + 0.5
        
      else:
        d = b    
      
      f2 = 1.0 - f1
      g2 = 1.0 - g1
 
      if ambig_groups[ag] > 1:
        ambig_matrix[a,b] += f1 * g1 * nobs
        if m > d >= 0:
          ambig_matrix[a,d] += f1 * g2 * nobs
        
        if n > c >= 0:
          ambig_matrix[c,b] += f2 * g1 * nobs
          if m > d >= 0:
            ambig_matrix[c,d] += f2 * g2 * nobs

      else:
        matrix[a,b] += f1 * g1 * nobs
        if  m > d >= 0:
          matrix[a,d] += f1 * g2 * nobs
        
        if n > c >= 0:
          matrix[c,b] += f2 * g1 * nobs
          if  m > d >= 0:
            matrix[c,d] += f2 * g2 * nobs
    
  else:
    for p_a, p_b, nobs, ag in contact_list:
      a = int((p_a-start_a)/bin_size)
      b = int((p_b-start_b)/bin_size)
 
      if ambig_groups[ag] > 1:
        ambig_matrix[a,b] += nobs
      else:
        matrix[a,b] += nobs

  
  if is_cis:
    matrix += matrix.T
    ambig_matrix += ambig_matrix.T
 
  return matrix, ambig_matrix


def get_region_array_matrix(contact_matrix, limits, region, orig_bin_size, bin_size):
  
  if bin_size != orig_bin_size:
    shape = _limits_to_shape(limits, limits, bin_size)
    contact_matrix = _downsample_matrix(contact_matrix, shape).astype(float)
  
  n, m =  contact_matrix.shape # n == m
  r1, r2 = region  
  start, end = limits
  shape = _limits_to_shape(region, region, bin_size)
  matrix = np.zeros(shape, float)
  
  a = int((r1-start)/bin_size)  # Requested index wrt stored data
  b = int(ceil((r2-start)/float(bin_size)))
  p = max(0, a) # Valid idx
  q = min(n, b)
  sub_contacts = contact_matrix[p:q,p:q]
  
  # Region could be outside stored data
  off = p-a
  lim = off + len(sub_contacts)  
  
  matrix[off:lim,off:lim] += sub_contacts
  matrix += matrix.T
  
  return matrix
  

def get_region_list_matrix(contact_list, region, bin_size, ambig_groups, smooth=False):
  
  start, end = region  
  start -= bin_size/2.0
  end += bin_size/2.0
  n, m = _limits_to_shape(region, (start, end), bin_size)
  
  matrix = np.zeros((n, m), float)
  ambig_matrix = np.zeros((n, m), float)
  
  if smooth:
    for p_a, p_b, nobs, ag in contact_list:
      if (start <= p_a <= end) and (start <= p_b <= end):
        p = (p_a-start)/bin_size
        q = (p_b-start)/bin_size
 
        a = int(p)
        b = int(q)
 
        f1 = p-a
        g1 = q-b
 
        if f1 > 0.5:
          c = a+1     # Alternative bin
          f1 = 1.5-f1 # Fraction in the main bin
 
        elif f1 < 0.5:
          c = a-1       # Alternative bin
          f1 = f1 + 0.5 # Fraction in the main bin
        else:
          c = a
 
        if g1 > 0.5:
          d = b+1
          g1 = 1.5-g1
 
        elif g1 < 0.5:
          d = b-1
          g1 = g1 + 0.5
 
        else:
          d = b
 
        f2 = 1.0 - f1
        g2 = 1.0 - g1
 
        if ambig_groups[ag] > 1:
          ambig_matrix[a,b] += f1 * g1 * nobs
          if m > d >= 0:
            ambig_matrix[a,d] += f1 * g2 * nobs
 
          if n > c >= 0:
            ambig_matrix[c,b] += f2 * g1 * nobs
            if m > d >= 0:
              ambig_matrix[c,d] += f2 * g2 * nobs

        else:
          matrix[a,b] += f1 * g1 * nobs
          if  m > d >= 0:
            matrix[a,d] += f1 * g2 * nobs
 
          if n > c >= 0:
            matrix[c,b] += f2 * g1 * nobs
            if  m > d >= 0:
              matrix[c,d] += f2 * g2 * nobs
            
  else:
    for p_a, p_b, nobs, ag in contact_list:
      if (start <= p_a <= end) and (start <= p_b <= end):
        a = int((p_a-start)/bin_size)
        b = int((p_b-start)/bin_size)
 
        if ambig_groups[ag] > 1:
          ambig_matrix[a,b] += nobs
 
        else:
          matrix[a,b] += nobs
        
  matrix += matrix.T
  ambig_matrix += ambig_matrix.T
 
  return matrix, ambig_matrix
  
  
def get_contact_lists_matrix(contacts, bin_size, chromos, chromo_limits):
    
  n, chromo_offsets, label_pos = _get_chromo_offsets(bin_size, chromos, chromo_limits)
  
  # Fill contact map matrix, last dim is for (un)ambigous
  matrix = np.zeros((n, n), float)
  ambig_matrix = np.zeros((n, n), float)
  
  ambig_groups = defaultdict(int)
    
  for key in contacts:
    for p_a, p_b, nobs, ag in contacts[key]:
      ambig_groups[ag] += 1

  trans_counts = {}
  homolog_groups = set()
  trans_groups = set()
  cis_groups = set()
  n_isol = 0
  n_pairs = 0
  
  for i, chr_1 in enumerate(chromos):
    for chr_2 in chromos[i:]:

      if chr_1 > chr_2:
        chr_a, chr_b = chr_2, chr_1
      else:
        chr_a, chr_b = chr_1, chr_2

      contact_list = contacts.get((chr_a, chr_b))

      if contact_list is None: # Nothing for this pair: common for single-cell Hi-C
        continue
      
      ni = 0
      isol = _get_isolated(contact_list)
      s_a, off_a, size_a = chromo_offsets[chr_a]
      s_b, off_b, size_b = chromo_offsets[chr_b]

      for p_a, p_b, nobs, ag in contact_list:
        if chr_a != chr_b:
          if ('.' in chr_a) and ('.' in chr_b) and (chr_a.split('.')[0] == chr_b.split('.')[0]):
            homolog_groups.add(ag)

          else:
            trans_groups.add(ag)

        else:
          cis_groups.add(ag)

        a = off_a + int((p_a-s_a)/bin_size)
        b = off_b + int((p_b-s_b)/bin_size)
  
 
        if ambig_groups[ag] == 1:
          matrix[a, b] += nobs
          matrix[b, a] += nobs
          n_pairs += 1
          
          if (p_a, p_b) in isol:
            ni += 1
  
        else:
          ambig_matrix[a, b] += nobs
          ambig_matrix[b, a] += nobs
      
      n_isol += ni
     
      if chr_a != chr_b:
        s1, e1 = chromo_limits[chr_a]
        s2, e2 = chromo_limits[chr_b]
        trans_counts[(chr_a, chr_b)] = (len(contact_list) - ni)/float((e1-s1) * (e2-s2))
        
  n_ambig = len([x for x in ambig_groups.values() if x > 1])
  n_homolog = len(homolog_groups)
  n_trans = len(trans_groups)
  n_cis = len(cis_groups)
  n_cont = len(ambig_groups)
  counts = (n_cont, n_cis, n_trans, n_homolog, n_ambig, n_pairs, n_isol)
  
  return counts, matrix, ambig_matrix, label_pos, chromo_offsets, trans_counts, ambig_groups


def _get_tick_delta(n_bins, bin_size_units, max_ticks=10):
  
  tick_delta = max(2, int(n_bins/float(max_ticks))) # Bins between ticks
  
  tick_delta_units = tick_delta * bin_size_units
  
  sf = int(floor(np.log10(tick_delta_units)))
  
  tick_delta_units = round(tick_delta_units, -sf) # round to nearest 100, 10, 1, 0.1 etc
  
  step = 10 ** sf
  target = 5 * max(1, sf)
  
  while (tick_delta_units % target > sf):
    tick_delta_units += step
  
  sf = int(floor(np.log10(tick_delta_units)))
  tick_delta_units = round(tick_delta_units, -sf)
  
  tick_delta = tick_delta_units/bin_size_units
  nminor = tick_delta # default to tick at each bin
  
  if nminor > max_ticks:
    nminor = 5
  
  return tick_delta, nminor


def get_diag_region(diag_width, matrix, double=False):

  n = len(matrix)
  d = min(n, diag_width+1) 
  
  if double:
    diag_mat = np.zeros((d+d-1, n*2))
    
    for y in range(1, d):
      rows = np.array(range(n-y))
      cols = rows + y
      xvals = rows + cols
      yvals = np.full(n-y, d+y-1) # d-1 is the mid point = diagonal
      counts = matrix[(rows, cols)]
      diag_mat[(yvals, xvals)] = counts
 
      xvals += 1
      diag_mat[(yvals[:-1], xvals[:-1])] = counts[:-1]
      
      if y:
        rows, cols = cols, rows
        yvals = np.full(n-y, d-1-y)
        counts = matrix[(rows, cols)]
        xvals -= 1
        diag_mat[(yvals, xvals)] = counts
 
        xvals += 1
        diag_mat[(yvals[:-1], xvals[:-1])] = counts[:-1]
 
      
  else:
    diag_mat = np.zeros((d, n*2))
    
    for y in range(d):
      rows = np.array(range(n-y))
      cols = rows + y
      xvals = rows + cols
      yvals = np.full(n-y, y)
      counts = matrix[(rows, cols)]
      diag_mat[(yvals, xvals)] = counts
 
      xvals += 1
      diag_mat[(yvals[:-1], xvals[:-1])] = counts[:-1]
  
  return diag_mat


def _get_trans_dev(trans_counts):

  cp = float(len(trans_counts))

  vals = np.array(list(trans_counts.values()), float)

  if not len(vals):
    return 0.0, '?'

  vals -= vals.min()
  vals /= vals.sum() or 1.0
  vals = vals[vals.argsort()]
  vals = vals.cumsum()

  base = np.arange(0, len(vals))/cp
  deltas = base - vals
  dev = 2.0 * deltas.sum()/cp

  if dev < 0.50:
    score_cat = '>4N'
  elif dev < 0.6:
    score_cat = '4N'
  elif dev < 0.74:
    score_cat = '2N'
  elif dev < 0.76:
    score_cat = '1/2N'
  elif dev < 0.95:
    score_cat = '1N'
  else:
    score_cat = '?'

  return dev, score_cat


def _get_mito_fraction(contacts, bin_size, min_sep=1e2, sep_range=(10**6.5, 10**7.5)):

  a, b = sep_range
  in_range = 0
  total = 0
  
  if bin_size:
    for chr_pair in contacts:
      chr_a, chr_b = chr_pair

      if chr_a != chr_b:
        continue
      
      contact_matrix = contacts[chr_pair]
      n, m = contact_matrix.shape
      n_cont = 0
      smaller = 0
      larger = 0
      
      for i in range(n-1):
        for j in range(i+1, n):
          sep = bin_size * (j-i)
          n_obs = contact_matrix[i,j]
          
          if sep <= a:
            smaller += n_obs
          elif sep >= b:
            larger += n_obs  
          
          n_cont += n_obs

      total += n_cont
      in_range += n_cont-(smaller+larger)
  
  else:
    for chr_pair in contacts:
      chr_a, chr_b = chr_pair

      if chr_a != chr_b:
        continue

      points = np.array(contacts[chr_pair])[:,:2]

      if len(points) < 3:
        continue

      d_seps = np.diff(points, axis=1)
      d_seps = d_seps[(d_seps > min_sep).nonzero()]

      n = len(d_seps)
      smaller = len((d_seps <= a).nonzero()[0])
      larger = len((d_seps >= b).nonzero()[0])

      total += n
      in_range += n-(smaller+larger)

  if not total:
    return 0.0, '?'

  frac = in_range/float(total)

  if frac < 0.30:
    score_cat = 'Non-M'
  elif frac < 0.40:
    score_cat = 'M'
  else:
    score_cat = 'Strong M'

  return frac, score_cat


def _get_isolated(positions, threshold=int(2e6)):

  isolated = set()
  bin_offsets = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))

  idx = defaultdict(list)

  for i, (p_a, p_b, nobs, ag) in enumerate(positions):
    idx[(p_a/threshold, p_b/threshold)].append(i)

  for key in idx:
    if len(idx[key]) == 1:
      pA, pB, nobs, ag = positions[idx[key][0]]
      b1, b2 = key

      for j, k in bin_offsets:
        key2 = (b1+j, b2+k)

        if key2 in idx:
          for i2 in idx[key2]:
            pC, pD, nobs2, ag2 = positions[i2]

            if abs(pC-pA) < threshold and abs(pD-pB) < threshold:
              break

            elif abs(pD-pA) < threshold and abs(pC-pB) < threshold:
              break

          else:
            continue

          break

      else:
        isolated.add((pA, pB))

  return isolated


def _is_detailed_data(data_track, step_size):
  
  return True
  
  m_width = np.median(np.abs(data_track['pos2']-data_track['pos1']))
  
  if m_width < 2 * step_size:
    return False
  
  else:
    return True
    
    
def plot_contact_matrix(matrix, bin_size, title, scale_label, chromo_labels=None,
                        axis_chromos=None, grid=None, stats_text=None, colors=None,
                        bad_color='#404040', x_data_tracks=None, y_data_tracks=None,
                        log=True, pdf=None, watermark='nuc_tools.contact_map',
                        legend=None, tracks=None, v_max=None, v_min=None,
                        ambig_matrix=None, diag_width=None, double_diag=False,
                        x_start=0, y_start=0):
  
  from nuc_tools import util
  mmax = matrix.max()
  if not mmax:
    util.warn('Map empty for ' + title, line_return=True)
    return
  
  if mmax < 0:
    matrix = -1 * matrix
  
  if colors and isinstance(colors, Colormap):
    cmap = colors
  
  else:
    if not colors:
      if log or (matrix.min() < 0):
        colors = ['#0000B0', '#0080FF', '#FFFFFF', '#FF0000', '#800000']
      else:
        colors = ['#FFFFFF', '#0080FF' ,'#FF0000','#000000']
  
    cmap = LinearSegmentedColormap.from_list(name='pcm', colors=colors, N=255)    
    cmap.set_bad(color=bad_color)

  if (ambig_matrix is not None) and ambig_matrix.max():
    do_ambig = True
    ambig_colors = ['#FFFFFF', '#D0D000', '#C0C000', '#B0B000']
    cmap2 = LinearSegmentedColormap.from_list(name='pcm', colors=ambig_colors, N=255)
    cmap2.set_bad(color=bad_color)
    
    clist = cmap(np.arange(cmap.N))
    clist[0,-1] = 0.0
    cmap = ListedColormap(clist)
    
  else:
    do_ambig = False  
 
  track_cmap = util.string_to_colormap(DT_COLORMAP)
   
  a, b = matrix.shape

  if diag_width and (a != b):
    util.critical('Diagonal width option only valid for square matrices. Input size was %d x %d' % (a,b))
  
  if bin_size * max(a,b) < 1e6:
    unit_name = 'kb'
    unit = 1e3
    label_pat = '%d'
  
  else:
    unit_name = 'Mb'
    unit = 1e6 
    label_pat = '%.f'
          
  if log:
    norm = LogNorm(vmin=v_min)
    v_min = None # No need for further clipping

  else:
    norm = None
    if v_max is None:
      v_max = max(-matrix.min(), matrix.max())
 
    if v_min is None:
      v_min = -v_max
      
  bg_color = np.array(cmap(0.0))
  dt_alpha = 0.5
    
  if diag_width:
    w = int(diag_width * unit/bin_size)
    nax = max(4, int(ceil(a/float(w))))
    diag_thick = (w-10*nax)/nax
    size = 8.0
     
    if double_diag:
      diag_thick /= 2
    
    aspect = 1.0  
    
    fig = plt.figure()
    fig.set_size_inches(size, size*nax/4.0)
    
    height = 0.8/float(nax)
    dt_frac = 0.2
    gap_frac = 0.2
    
    kw = {'interpolation':'None', 'norm':norm, 'origin':'lower',
          'vmin':v_min, 'vmax':v_max}
    
    if do_ambig:
      diag_mat_ambig = get_diag_region(diag_thick, ambig_matrix, double_diag)
    else:
      diag_mat_ambig = None

    diag_mat = get_diag_region(diag_thick, matrix, double_diag)
    tick_delta, nminor = _get_tick_delta(w, bin_size/unit)
        
    for i in range(nax):
      xminor_tick_locator = AutoMinorLocator(nminor)
    
      p1 = i*w*2
      
      if p1 >= 2*a:
        ax.set_visible(False)
        continue
      
      p2 = min(2*a, (i+1)*w*2)
      
      w_frac = (p2-p1)/(2.0*w)
      row_bottom = 0.1 + (nax-i-1)*height
         
      if x_data_tracks:
        map_frac = 1.0-dt_frac-gap_frac
        
        if double_diag:
          ax = fig.add_axes([0.1, row_bottom + ((1.0-0.5*map_frac) * height),0.8*w_frac, map_frac*height*0.5])
          ax_dt = fig.add_axes([0.1, row_bottom + ((gap_frac+0.5*map_frac) * height), 0.8*w_frac, dt_frac*height])
          ax2 = fig.add_axes([0.1, row_bottom + (gap_frac * height),0.8*w_frac, map_frac*height*0.5])
          ax_bott = ax2
          
        else:
          ax = fig.add_axes([0.1, row_bottom + ((dt_frac+gap_frac) * height), 0.8*w_frac, map_frac*height])
          ax_dt = fig.add_axes([0.1, row_bottom + (gap_frac * height), 0.8*w_frac, dt_frac*height])
          ax_bott = ax_dt
          ax2 = None
          
      else:
        ax = fig.add_axes([0.1, row_bottom + (gap_frac * height), 0.8*w_frac, (1.0-gap_frac)*height])
        ax_bott = ax
        ax_dt = None
        ax2 = None
        
      if ax2:
        hw = int((diag_mat.shape[0]-1)//2)
      
        if diag_mat_ambig is not None:
          ax.matshow(diag_mat_ambig[-hw:,p1:p2], cmap=cmap2, aspect='auto', **kw)
          ax2.matshow(diag_mat_ambig[:hw,p1:p2], cmap=cmap2, aspect='auto', **kw)
 
        cax = ax.matshow(diag_mat[-hw:,p1:p2], cmap=cmap, aspect='auto', **kw)
        cax2 = ax2.matshow(diag_mat[:hw,p1:p2], cmap=cmap, aspect='auto', **kw)
      
      else:
        if diag_mat_ambig is not None:
          ax.matshow(diag_mat_ambig[:,p1:p2], cmap=cmap2, aspect='auto', **kw)
 
        cax = ax.matshow(diag_mat[:,p1:p2], cmap=cmap, aspect='auto', **kw)
      
      if x_data_tracks:
        nx = float(len(x_data_tracks))
        track_start = p1*0.5*bin_size+x_start
        track_end = p2*0.5*bin_size+x_start
        
        step = bin_size
        while (track_end-track_start)/step < 500:
          step /= 2
        
        nx = float(len(x_data_tracks))
        colors = [np.array(track_cmap(x)) for x in np.linspace(0.0, 1.0, nx)]
        min_width = 2.0 * size/float(b)
 
        for j, (track_label, track_data) in enumerate(x_data_tracks):          
          t = nx-j-1.0
   
          track_data = track_data[(track_data['pos1'] < track_end) & (track_data['pos1'] > track_start) | \
                                  (track_data['pos2'] < track_end) & (track_data['pos2'] > track_start)]
          pos_strand = track_data['strand']
          pos_track = track_data[pos_strand]
          neg_track = track_data[~pos_strand]
          tcmap = LinearSegmentedColormap.from_list(name=track_label, colors=[bg_color, colors[j]], N=32)

          if _is_detailed_data(track_data, step):
            if len(pos_track):
              starts = (pos_track['pos1']-track_start)/float(0.5*bin_size)
              ends = (pos_track['pos2']-track_start)/float(0.5*bin_size)
              ends = np.array([ends, starts+min_width]).max(axis=0)
              widths = ends - starts
              values = pos_track['value']
              heights = heights*values
              
              if len(neg_track):
                height =  0.5/nx
                y_pos = np.full(widths.shape, (t+0.75)/nx)
                y_pos += heights/2.0
                
              else:
                height =  1.0/nx
                y_pos = np.full(widths.shape, (t+0.5)/nx)
              
              vcolors = values[:,None] * 0.75 + 0.25
              vcolors = np.clip((vcolors * colors[i]) + ((1.0-vcolors) * bg_color), 0.0, 1.0)
              ax_dt.barh(y_pos, widths, heights, starts,
                         color=vcolors, linewidth=0.0)
                           
          
            if len(neg_track):
              starts = (neg_track['pos1']-x_start)/float(0.5*bin_size)
              ends = (neg_track['pos2']-x_start)/float(0.5*bin_size)
              ends = np.array([ends, starts+min_width]).max(axis=0)

              widths = ends - starts
              values = neg_track['value']
              heights = heights*values
 
              if len(pos_track):
                height =  0.5/nx
                y_pos = np.full(widths.shape, (t+0.25)/nx)
                y_pos -= heights/2.0
 
              else:
                height =  1.0/nx
                y_pos = np.full(widths.shape, (t+0.5)/nx)
 
              vcolors = values[:,None] * 0.75 + 0.25
              vcolors = np.clip((vcolors * colors[i]) + ((1.0-vcolors) * bg_color), 0.0, 1.0)
              ax_dt.barh(y_pos, widths, heights, starts,
                         color=vcolors, linewidth=0.0)

          else:
            if len(pos_track):
              pos_hist = util.bin_data_track(pos_track, step, track_start, track_end)[None,:]
 
              if len(neg_track):
                extent=(0,p2-p1,(t+0.5)/nx,(t+1.0)/nx)
              else:
                extent=(0,p2-p1,t/nx,(t+1.0)/nx)
 
              ax_dt.matshow(pos_hist, aspect='auto', cmap=tcmap, extent=extent)
              
            if len(neg_track):
              neg_hist = util.bin_data_track(neg_track, step, track_start, track_end)[None,:]
 
              if len(pos_track):
                extent=(0,p2-p1,t/nx,(t+0.5)/nx)
              else:
                extent=(0,p2-p1,t/nx,(t+1.0)/nx)
 
              ax_dt.matshow(neg_hist, aspect='auto', cmap=tcmap, extent=extent)

        ax_dt.set_ylim(0.0, 1.0)
        ax_dt.set_facecolor(cmap(0.0))

        ax.tick_params(which='both', direction='out',
                       left=False, right=False, top=False, bottom=False,
                       labelright=False, labelleft=False, labeltop=False, labelbottom=False)
        
        if ax2:
          ax_dt.tick_params(which='both', direction='out',
                            left=False, right=False, top=False, bottom=False,
                            labelright=False, labelleft=False, labeltop=False, labelbottom=False)
                
        ax_bott.tick_params(which='both', direction='out', labelsize=9,
                            left=False, right=False, top=False, bottom=True,
                            labelright=False, labelleft=False, labeltop=False, labelbottom=True)
        
      else:
        ax_bott = ax
        ax.tick_params(which='both', direction='out', labelsize=9,
                       left=False, right=False, top=False, bottom=True,
                       labelright=False, labelleft=False, labeltop=False, labelbottom=True)
                
      ax_bott.set_anchor('W') 
      
      if tick_delta < (p2-p1):
        xlabel_pos = np.arange(p1, p2+1, tick_delta) # Pixel bins
        xlabels = ['%.2f' % ((x*0.5*bin_size+x_start)/unit) for x in xlabel_pos]
        xlabel_pos -= p1

        ax_bott.set_xticklabels(xlabels, fontsize=9)
        ax_bott.xaxis.set_ticks(xlabel_pos)
        ax_bott.xaxis.set_minor_locator(xminor_tick_locator)
        
      ax_bott.set_xlim(0, p2-p1)
    
    ax_bott.set_xlabel('Position (%s)' % unit_name, fontsize=12)
       
    if x_data_tracks: 
      legend_ax = fig.add_axes([0.1, 0.0, 0.8, 0.095])
      for i, (track_label, track_data) in enumerate(x_data_tracks):
        legend_ax.plot([], alpha=dt_alpha, label=track_label, color=colors[i])
 
      legend_ax.set_axis_off()
      legend_ax.legend(frameon=False, loc='upper left', fontsize=7, ncol=int(math.ceil(len(x_data_tracks)/3.0)))
    
    if do_ambig:
      cbaxes = fig.add_axes([0.92, 0.15, 0.02, 0.3])
      cbar2 = plt.colorbar(cax2, cax=cbaxes)
      cbar2.ax.tick_params(labelsize=7)
      cbar2.set_label('Ambig. count', fontsize=7)
      _clean_log_ticks(cbar2.ax)
    
    cbaxes = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    cbar = plt.colorbar(cax, cax=cbaxes)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(scale_label, fontsize=7)
    _clean_log_ticks(cbar.ax)

    dpi= int(float(w)/(fig.get_size_inches()[1]*ax.get_position().size[1]))
          
  else:
 
    if chromo_labels:
      xlabel_pos, xlabels = zip(*chromo_labels)
      ylabel_pos = xlabel_pos
      ylabels = xlabels
      xrotation = 90.0
      xminor_tick_locator = None
      yminor_tick_locator = None
 
    else:
      xrotation = None
      tick_delta, nminor = _get_tick_delta(b, bin_size/unit)
      xlabel_pos = np.arange(0, b, tick_delta) # Pixel bins      
      xlabels = [label_pat % ((x*bin_size+x_start)/unit) for x in xlabel_pos]
      xminor_tick_locator = AutoMinorLocator(nminor)
 
      tick_delta, nminor = _get_tick_delta(a, bin_size/unit)
      ylabel_pos = np.arange(0, a, tick_delta) # Pixel bins
      ylabels = [label_pat % ((y*bin_size+y_start)/unit) for y in ylabel_pos]
      yminor_tick_locator = AutoMinorLocator(nminor)
    
    fig = plt.figure()
    
    dt_size = 0.8
    padd = 0.1
    size = 8.0
    main_frac = 0.8
    
    if x_data_tracks:
      if y_data_tracks:
        fig.set_size_inches(size + dt_size, size + dt_size)
        fx = fy = dt_size / (size + dt_size)
        ax = fig.add_axes([padd + fx, padd + fy, main_frac * (1.0-fx), main_frac * (1.0-fy)]) # left, bottom, width, height
        ax_left = fig.add_axes([padd, padd + fy, fx, main_frac * (1.0-fy)])
        ax_bott = fig.add_axes([padd + fx, padd, main_frac * (1.0-fx), fy])
        ax_left.set_facecolor('w')
        ax_bott.set_facecolor('w')
        
      else:
        fig.set_size_inches(size + dt_size, size)
        fx = dt_size / (size + dt_size)
        ax = fig.add_axes([padd + fx, padd, main_frac * (1.0-fx), main_frac]) # left, bottom, width, height
        ax_left = ax
        ax_bott = fig.add_axes([padd + fx, padd, main_frac * (1.0-fx), fy])
        ax_bott.set_facecolor('w')
         
    elif y_data_tracks:
      fig.set_size_inches(size, size + dt_size)
      fy = dt_size / (size + dt_size)
      ax = fig.add_axes([padd, padd + fy, main_frac, main_frac * (1.0-fy)]) # left, bottom, width, height
      ax_left = fig.add_axes([padd, padd + fy, fx, main_frac * (1.0-fy)])
      ax_left.set_facecolor('w')
      ax_bott = ax
      
    else:
      fig.set_size_inches(size, size)
      ax = ax_left = ax_bott = fig.add_axes([padd, padd, main_frac, main_frac]) # left, bottom, width, height
    
    if grid and grid is not True:
      grid = np.array(grid, float)
      ax.hlines(grid, 0.0, float(b), color='#B0B0B0', alpha=0.5, linewidth=0.1)
      ax.vlines(grid, float(a), 0.0, color='#B0B0B0', alpha=0.5, linewidth=0.1)
      
    kw = {'interpolation':'None', 'norm':norm, 'origin':'upper',
          'vmin':v_min, 'vmax':v_max}
    
    extent = (0,b,a,0)
    
    gff_cdict = {'CDS':(0.0, 0.0, 0.0, 1.0)}
    gff_vdict = {'gene': 0.15, 'exon': 0.5, 'CDS': 0.85}
    
    if do_ambig:
      cax2 = ax.matshow(ambig_matrix, cmap=cmap2, extent=extent, **kw)
 
    cax = ax.matshow(matrix, cmap=cmap, extent=extent, **kw)
     
    if x_data_tracks:
      start = x_start
      end = start + b*bin_size
      
      step = bin_size
      while (end-start)/step < 500:
        step /= 2
      
      y_lim = [0.0, 1.0]
      nx = len(x_data_tracks)
      colors = [np.array(track_cmap(x)) for x in np.linspace(0.0, 1.0, nx)]
      nx = float(nx)
      min_width = 1.0 * size/float(b)
            
      for i, (track_label, track_data) in enumerate(x_data_tracks):
        t = nx-i-1.0
        y_anchor = (t+0.5)/nx
        track_data = track_data[(track_data['pos1'] < end) & (track_data['pos1'] > start) | (track_data['pos2'] < end) & (track_data['pos2'] > start)]
        pos_strand = track_data['strand']
        pos_track = track_data[pos_strand]
        neg_track = track_data[~pos_strand]
        tcmap = LinearSegmentedColormap.from_list(name=track_label, colors=[bg_color, colors[i]], N=32)
        
        ax_bott.hlines(y_anchor, start, end, colors='#808080', alpha=0.2, linewidth=0.1)
        
        if _is_detailed_data(track_data, step):
          if len(pos_track):
            starts = (pos_track['pos1']-x_start)/float(bin_size)
            ends = (pos_track['pos2']-x_start)/float(bin_size)
            ends = np.array([ends, starts+min_width]).max(axis=0)
            widths = ends - starts
            labels = pos_track['label'].astype(str)
            
            if ';' in labels[0]: # GFF/GTF
              features = [itm.split(';')[0] for itm in labels]
              vcolors = np.array([gff_cdict.get(f, colors[i]) for f in features])
              values = np.array([gff_vdict.get(f, 0.5) for f in features])
              idx = values.argsort()[::-1] # smallest last (on top)
              widths = widths[idx]
              values = values[idx]
              vcolors = vcolors[idx]
              starts = starts[idx]
            
            else:
              values = pos_track['value']
              vcolors = values[:,None] * 0.75 + 0.25
              vcolors = np.clip((vcolors * colors[i]) + ((1.0-vcolors) * bg_color), 0.0, 1.0)
              
            y_pos = np.full(widths.shape, y_anchor)            
            
            if len(neg_track):           
              height =  0.5/nx
              heights = height*values
              y_pos += heights/2.0
              
            else:
              height =  1.0/nx
              heights = height*values
            
            ax_bott.barh(y_pos, widths, heights, starts,
                         color=vcolors, linewidth=0.0)
          
          if len(neg_track):
            starts = (neg_track['pos1']-x_start)/float(bin_size)
            ends = (neg_track['pos2']-x_start)/float(bin_size)
            ends = np.array([ends, starts+min_width]).max(axis=0)
            widths = ends - starts
            labels = neg_track['label'].astype(str)

            if ';' in labels[0]: # GFF/GTF
              features = [itm.split(';')[0] for itm in labels]
              vcolors = np.array([gff_cdict.get(f, colors[i]) for f in features])
              values = np.array([gff_vdict.get(f, 0.5) for f in features])
              idx = values.argsort()[::-1] # smallest last (on top)
              widths = widths[idx]
              values = values[idx]
              vcolors = vcolors[idx]
              starts = starts[idx]
              
            else:
              values = neg_track['value']
              vcolors = values[:,None] * 0.75 + 0.25
              vcolors = np.clip((vcolors * colors[i]) + ((1.0-vcolors) * bg_color), 0.0, 1.0)
            
            y_pos = np.full(widths.shape, y_anchor)            
            
            if len(pos_track):           
              height =  0.5/nx
              heights = height*values
              y_pos -= heights/2.0
             
            else:
              height =  1.0/nx
              heights = height*values
            
            ax_bott.barh(y_pos, widths, heights, starts,
                         color=vcolors, linewidth=0.0)

        else:          
          if len(pos_track):
            pos_hist = util.bin_data_track(pos_track, step, start, end)[None,:]
            
            if len(neg_track):
              extent=(0,b,(t+0.5)/nx,(t+1.0)/nx)
            else:
              extent=(0,b,t/nx,(t+1.0)/nx)
              
            ax_bott.matshow(pos_hist, aspect='auto', cmap=tcmap, extent=extent)
          
          if len(neg_track):
            neg_hist = util.bin_data_track(neg_track, step, start, end)[None,:]
            
            if len(pos_track):
              extent=(0,b,t/nx,(t+0.5)/nx)
            else:
              extent=(0,b,t/nx,(t+1.0)/nx)
            
            ax_bott.matshow(neg_hist, aspect='auto', cmap=tcmap, extent=extent)

      ax_bott.set_ylim(*y_lim)
      #ax_bott.set_facecolor(cmap(0.0))
      
      x0, y0, w, h = ax_bott.get_position().bounds
      
      legend_ax = fig.add_axes([2*padd, 0.0, w, y0])
      for i, (track_label, track_data) in enumerate(x_data_tracks):
        legend_ax.plot([], alpha=dt_alpha, label=track_label, color=colors[i])
      
      legend_ax.set_axis_off()
      legend_ax.legend(frameon=False, handlelength=1.0, columnspacing=1.0, handletextpad=0.5, borderpad=0.2,
                       loc='lower left', fontsize=7, ncol=min(7, len(x_data_tracks)))
      
      ax.set_xticks([])
      ax_bott.set_yticks([])
      
    if y_data_tracks:
      start = y_start
      end = start + a*bin_size
      
      step = bin_size
      while (end-start)/step < 500:
        step /= 2
      
      x_lim = [0.0, 1.0]
      ny = len(y_data_tracks)
      
      colors = [np.array(track_cmap(y)) for y in np.linspace(0.0, 1.0, ny)]
      ny = float(ny)
      
      for i, (track_label, track_data) in enumerate(y_data_tracks):
        t = ny-i-1.0
        x_anchor = (t+0.5)/ny
        
        track_data = track_data[(track_data['pos1'] < end) & (track_data['pos1'] > start) | (track_data['pos2'] < end) & (track_data['pos2'] > start)]
        pos_strand = track_data['strand']
        pos_track = track_data[pos_strand]
        neg_track = track_data[~pos_strand]
        tcmap = LinearSegmentedColormap.from_list(name=track_label, colors=[bg_color, colors[i]], N=32)
       
        ax_left.vlines(x_anchor, start, end, colors='#808080', alpha=0.2, linewidth=0.1)
       
        if _is_detailed_data(track_data, step):
          if len(pos_track):
            starts = (pos_track['pos1']-y_start)/float(bin_size)
            ends = (pos_track['pos2']-y_start)/float(bin_size)
            ends = np.array([ends, starts+min_width]).max(axis=0)
            heights = ends - starts
            labels = pos_track['label'].astype(str)
            x_pos = np.full(heights.shape, x_anchor)
            
            if ';' in str(labels[0]): # GFF/GTF
              features = [itm.split(';')[0] for itm in labels]
              vcolors = np.array([gff_cdict.get(f, colors[i]) for f in features])
              values = np.array([gff_vdict.get(f, 0.5) for f in features])
              idx = values.argsort()[::-1] # smallest last (on top)
              heights = heights[idx]
              values = values[idx]
              vcolors = vcolors[idx]
              starts = starts[idx]
            
            else:
              values = pos_track['value']
              vcolors = values[:,None] * 0.75 + 0.25
              vcolors = np.clip((vcolors * colors[i]) + ((1.0-vcolors) * bg_color), 0.0, 1.0)
            
            if len(neg_track):           
              width =  0.5/ny
              widths = width*values
              x_pos += widths/2.0
              
            else:
              width =  1.0/ny
              widths = width*values
                        
            ax_left.bar(x_pos, heights, widths, starts,
                        color=vcolors, linewidth=0.0)
          
          if len(neg_track):
            starts = (neg_track['pos1']-y_start)/float(bin_size)
            ends = (neg_track['pos2']-y_start)/float(bin_size)
            ends = np.array([ends, starts+min_width]).max(axis=0)
            heights = ends - starts   
            labels = neg_track['label'].astype(str)
            x_pos = np.full(heights.shape, x_anchor)
            
            if ';' in labels[0]: # GFF/GTF
              features = [itm.split(';')[0] for itm in labels]
              vcolors = np.array([gff_cdict.get(f, colors[i]) for f in features])
              values = np.array([gff_vdict.get(f, 0.5) for f in features])
              idx = values.argsort()[::-1] # smallest last (on top)
              heights = heights[idx]
              values = values[idx]
              vcolors = vcolors[idx]
              starts = starts[idx]
            
            else:
              values = neg_track['value']
              vcolors = values[:,None] * 0.75 + 0.25
              vcolors = np.clip((vcolors * colors[i]) + ((1.0-vcolors) * bg_color), 0.0, 1.0)
                    
            if len(pos_track):           
              width =  0.5/ny
              widths = width*values
              x_pos -= widths/2.0
            
            else:
              width =  1.0/ny
              widths = width*values
            
            ax_left.bar(x_pos, heights, widths, starts,
                        color=vcolors, linewidth=0.0)
        
        else:
          if len(pos_track):
            pos_hist = util.bin_data_track(pos_track, step, start, end)[::-1,None]
            
            if len(neg_track):
              extent=((t+0.5)/nx,(t+1.0)/nx,0,a)
            else:
              extent=(t/nx,(t+1.0)/nx,0,a)
            
            ax_left.matshow(pos_hist, aspect='auto', cmap=tcmap, extent=extent)
           
          if len(neg_track):
            neg_hist = util.bin_data_track(neg_track, step, start, end)[::-1,None]
            
            if len(pos_track):
              extent=(t/nx,(t+0.5)/nx,0,a)
            else:
              extent=(t/nx,(t+1.0)/nx,0,a)
           
            ax_left.matshow(neg_hist, aspect='auto', cmap=tcmap, extent=extent)
        
      ax_left.set_xlim(*x_lim)
      ax.set_yticks([])
      ax_left.set_xticks([])
      #ax_left.set_facecolor(cmap(0.0))
    
    
    if chromo_labels and len(xlabels) > 25:
      ax_bott.set_xticklabels(xlabels, fontsize=5, rotation=xrotation)
      ax_left.set_yticklabels(ylabels, fontsize=5)
 
    else:
      ax_bott.set_xticklabels(xlabels, fontsize=9, rotation=xrotation)
      ax_left.set_yticklabels(ylabels, fontsize=9)
      
    if x_data_tracks or y_data_tracks:
      ax_left.tick_params(which='both', direction='out', left=True, right=False, labelright=False, labelleft=True, 
                          labeltop=False, labelbottom=False, top=False, bottom=False, pad=8)
      ax_bott.tick_params(which='both', direction='out', labeltop=False, labelbottom=True, top=False, bottom=True,
                          left=False, right=False, labelright=False, labelleft=False, pad=8)
    
    else:
      ax.tick_params(which='both', direction='out', left=True, right=False, labelright=False, labelleft=True,
                     labeltop=False, labelbottom=True, top=False, bottom=True, pad=8)

    ax_left.yaxis.set_ticks(ylabel_pos)
    ax_left.set_ylim(a, 0)
    
    ax_bott.xaxis.set_ticks(xlabel_pos)
    ax_bott.set_xlim(0, b)
 
    if chromo_labels:
      ax_bott.set_xlabel('Chromosome')
      ax_left.set_ylabel('Chromosome')
 
    elif axis_chromos:
      ax_left.set_ylabel('Position %s (%s)' % (axis_chromos[0], unit_name))
      ax_bott.xaxis.set_minor_locator(xminor_tick_locator)
      ax_bott.set_xlabel('Position %s (%s)' % (axis_chromos[1], unit_name))
      ax_left.yaxis.set_minor_locator(yminor_tick_locator)
      if grid is True and not log:
        ax.grid(alpha=0.08, linestyle='-', linewidth=0.1)
 
    else:
      ax_bott.set_xlabel('Position (%s)' % unit_name)
      ax_bott.xaxis.set_minor_locator(xminor_tick_locator)
      ax_left.set_ylabel('Position (%s)' % unit_name)
      ax_left.yaxis.set_minor_locator(yminor_tick_locator)
 
      if grid is True and not log:
        ax.grid(alpha=0.08, linestyle='-', linewidth=0.1)
 
    if do_ambig:
      cbaxes = fig.add_axes([0.915, 0.2, 0.02, 0.3])
      cbar2 = plt.colorbar(cax2, cax=cbaxes)
      cbar2.ax.tick_params(labelsize=8)
      cbar2.set_label('Ambig. count', fontsize=8)
      _clean_log_ticks(cbar2.ax)
 
    cbaxes = fig.add_axes([0.915, 0.6, 0.02, 0.3])
    cbar = plt.colorbar(cax, cax=cbaxes)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(scale_label, fontsize=8)
    _clean_log_ticks(cbar.ax)
  
    dpi= int(float(a)/(fig.get_size_inches()[1]*ax.get_position().size[1]))
            
  if stats_text:
    ax.text(0.9, 0.92, stats_text, fontsize=9, transform=fig.transFigure, ha='right')  
    
  ax.text(0.01, 0.01, watermark, color='#B0B0B0', fontsize=8, transform=fig.transFigure) 
  ax.text(0.1, 0.95, title, color='#000000', fontsize=12, transform=fig.transFigure, ha='left') 
  #ax.set_title(title, loc='left')
 
  if legend:
    for label, color in legend:
      ax.plot([], linewidth=3, label=label, color=color)
    
    ax.legend(fontsize=8, loc=9, ncol=len(legend), bbox_to_anchor=(0.5, 1.05), frameon=False)
    
  dpi = max(10, dpi)
  
  while dpi < 300:
    dpi *= 2
  
  util.info(' .. making map ' + title + ' (dpi=%d)' % dpi, line_return=True)

  if pdf:
    pdf.savefig(dpi=dpi)
  else:
    plt.show()
    
  plt.close()


def _clean_log_ticks(ax):
  
  tick_labels = ax.get_yticklabels()
      
  for i, label in enumerate(tick_labels):
    text = label.get_text()
    
    if '\\times10^{0}' in text:
      text = text.replace('\\times10^{0}', '')
    elif '10^{0}' in text:
      text = text.replace('10^{0}', '1')
    elif '10^{1}' in text:
      text = text.replace('10^{1}', '10')
    else:
      continue
       
    label.set_text(text)
  
  ax.set_yticklabels(tick_labels)


def _get_vmax(matrix):  
 
  diag = np.diag(matrix)
  diag_nz = diag[diag.nonzero()]
  
  if len(diag_nz):
    return np.percentile(diag_nz, 95.0)
    
  else:
    return 1.0
      
                   
def contact_map(in_paths, out_path, bin_size=None, bin_size2=250.0, bin_size3=500.0,
                no_separate_cis=False, separate_trans=False, show_chromos=None,
                region_dict=None, use_corr=False, use_norm=False, is_single_cell=False,
                screen_gfx=False, black_bg=False, min_contig_size=None, chromo_grid=False,
                diag_width=None, data_tracks=None, gff_feats=None,
                smooth=False, font=None, font_size=12, line_width=0.2, cmap=None):
  
  # Data tracks are specified as (file_path, file_format, text_label)
  
  from nuc_tools import io, util
  from formats import ncc, npz, gff, bed, wig, sam
  
  if len(in_paths) == 2:
    in_path, in_path2 = in_paths
    in_msg = '%s and %s' % tuple(in_paths)
  else:
    in_path = in_paths[0]
    in_path2 = None
    in_msg = in_path
    
  if out_path:
    out_path = io.check_file_ext(out_path, '.pdf')
  
  elif in_path2:
    if region_dict:
      suffix='_regions_comb'
    else:
      suffix='_comb'
  
    out_path = io.check_file_ext(io.merge_file_names(in_path, in_path2, suffix=suffix), '.pdf')
    
  else:
    if region_dict:
      out_path = io.check_file_ext(os.path.splitext(in_path)[0] + '_regions', '.pdf')
    else:
      out_path = io.check_file_ext(in_path, '.pdf')
  
  if data_tracks:
    file_paths, file_formats, text_labels = zip(*data_tracks)
    default_labels = io.check_file_labels(list(text_labels), file_paths)
  
    for i, (file_path, file_format, text_label) in enumerate(data_tracks):
      if not text_label:
        data_tracks[i] = (file_path, file_format, default_labels[i])
  else:
    data_tracks = []
          
  if screen_gfx:
    util.info('Displaying contact map for {}'.format(in_msg))
  else:
    util.info('Making PDF contact map for {}'.format(in_msg))
  
  if io.is_ncc(in_path):
    file_bin_size = None
    util.info('Loading NCC format contact data')
    chromosomes, chromo_limits, contacts = ncc.load_file(in_path)
    
  else:
    smooth = False
    util.info('Loading NPZ format contact data')
    file_bin_size, chromo_limits, contacts = npz.load_npz_contacts(in_path)
      
    #normalize_contacts(contacts, chromo_limits, file_bin_size, store_sparse=False)

  if in_path2:
  
    if io.is_ncc(in_path2):
      file_bin_size2 = None
      chromosomes2, chromo_limits2, contacts2 = ncc.load_file(in_path2)
 
    else:
      smooth = False
      file_bin_size2, chromo_limits2, contacts2 = npz.load_npz_contacts(in_path2)
      
      if file_bin_size and (file_bin_size2 != file_bin_size):
        util.critical('Input contact datsets are binned at different resolutions')
  else:
    contacts2 = None
      
  if not chromo_limits:
    util.critical('No chromosome contact data read')

  if min_contig_size:
    min_contig_size = int(min_contig_size * 1e6)
  else:
    largest = max([e-s for s, e in chromo_limits.values()])
    min_contig_size = int(DEFAULT_SMALLEST_CONTIG*largest) 
    
    if not region_dict:
      msg = 'Min. contig size not specified, using {}% of largest: {:,} bp'
      util.info(msg.format(DEFAULT_SMALLEST_CONTIG*100, min_contig_size))
  
  if region_dict:
    chr_names = ', '.join(sorted(chromo_limits))
    
    for chromo in sorted(region_dict):
      if chromo not in chromo_limits:
        if chromo.lower().startswith('chr') and (chromo[3:] in chromo_limits):
          region_dict[chromo[3:]] = region_dict[chromo]
          del region_dict[chromo]
        
        else:
          msg = 'Chromosome "%s" doesn\'t match any in the contact file. Available: {}'
          util.critical(msg.format(chromo, chr_names))
    
    chromo_limits = {chromo:chromo_limits[chromo] for chromo in region_dict}
    
    if not chromo_limits:
      msg = 'No chromosomes or regions match the contact file. Available: {}'
      util.critical(msg.format(chr_names))
    
  elif show_chromos:
    chr_names = ', '.join(sorted(chromo_limits))
    
    filtered = {}
    found = set()
    for chromo, lims in chromo_limits.items():
      if chromo in show_chromos:
        filtered[chromo] = lims
        found.add(chromo)
        
      elif chromo.lower().startswith('chr') and (chromo[3:] in show_chromos):
        filtered[chromo] = lims
        found.add(chromo[3:])
    
    unknown = sorted(set(show_chromos) - found)
         
    chromo_limits = filtered
  
    if not chromo_limits:
      util.critical('Chromosome selection doesn\'t match any in the contact file. Available: {}'.format(chr_names))
    elif unknown:
      util.warn('Some selected chromosomes don\'t match the contact file: {}'.format(', '.join(unknown)))
  
  def _adapt_bin_size(bsize, sq_len, min_bin_count):
    
    n_bins = sq_len/bsize
    while n_bins < min_bin_count:
      bsize /= 2.0
      bsize = round(bsize, -int(math.floor(math.log10(abs(bsize)))))
      n_bins = sq_len/bsize    
    
    return bsize
    
  if bin_size:
    bin_size = int(bin_size * 1e3)
    
    if file_bin_size and (bin_size < file_bin_size):
      bin_size = file_bin_size
      msg = 'Smallest main display bin size limited to innate resolution of input data'
      util.warn(msg)
      
  else:
    seq_len = sum([e-s for s, e in chromo_limits.values() if e-s > min_contig_size])
    
    bin_size = DEFAULT_SC_MAIN_BIN_KB if is_single_cell else DEFAULT_MAIN_BIN_KB
    bin_size *= 1e3    
    bin_size = _adapt_bin_size(bin_size, seq_len, 1000)
  
  if bin_size2:
    bin_size2 = int(bin_size2 * 1e3)
    
    if file_bin_size and (bin_size2 < file_bin_size):
      bin_size2 = file_bin_size
      msg = 'Smallest cis display bin size limited to innate resolution of input data'
      util.warn(msg)
  
  else:
    bin_size2 = DEFAULT_SC_CHR_BIN_KB if is_single_cell else DEFAULT_CIS_BIN_KB
    bin_size2 *= 1e3
    bin_size2 = _adapt_bin_size(bin_size2, min_contig_size, 10)
    
  if bin_size3:
    bin_size3 = int(bin_size3 * 1e3)
    
    if file_bin_size and (bin_size3 < file_bin_size):
      bin_size3 = file_bin_size
      msg = 'Smallest main display bin size limited to innate resolution of input data'
      util.warn(msg)
  
  else:
    bin_size3 = bin_size2 if is_single_cell else 2*bin_size2

  """
  tot_size = 0
  
  for chromo in chromo_limits:
    s, e = chromo_limits[chromo]
    size = e-s
    
    if size >= min_contig_size:
      tot_size += size 
  
  bin_size = int(tot_size/1000)
  util.info('Bin size not specified, using approx. 1000 x 1000 bin equivalent: {:,} bp'.format(bin_size))
  """
    
  separate_cis = not bool(no_separate_cis)
          
  # Get sorted chromosomes, ignore small contigs as appropriate
  chromos = []
  skipped = []
  
  if region_dict: # No filtering on size if region specified
    chromos = region_dict.keys()
    
  else:
    for chromo in chromo_limits:
      s, e = chromo_limits[chromo]

      if (e-s) < min_contig_size:
        if show_chromos and (chromo in show_chromos):
          msg = 'Chromosome {} is below the size limit but was nonethless included as it was included in the -chr option'
          util.info(msg.format(chromo))
        else:
          skipped.append(chromo)
          continue

      chromos.append(chromo)
      
    util.info('Considering {:,} chromosomes/contigs'.format(len(chromos)))
    
  if skipped:
    util.info('Skipped {:,} small chromosomes/contigs < {:,} bp'.format(len(skipped), min_contig_size))

  chromos = util.sort_chromosomes(chromos)
   
  chromo_labels = []
  for chromo in chromos:
    if chromo.upper().startswith('CHR'):
      chromo = chromo[3:]
    chromo_labels.append(chromo)

  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(out_path)
      
  if file_bin_size:
    ret = get_contact_arrays_matrix(contacts, bin_size, chromos, chromo_limits)
    count_list, full_matrix, label_pos, offsets = ret
    n_cont, n_cis, n_trans, n_homolog, n_ambig = count_list
    ambig_matrix = None
    n_isol = None
    trans_counts = None

  else:
    ret = get_contact_lists_matrix(contacts, bin_size, chromos, chromo_limits)
    count_list, full_matrix, ambig_matrix, label_pos, offsets, trans_counts, ambig_groups = ret
    n_cont, n_cis, n_trans, n_homolog, n_ambig, n_pairs, n_isol = count_list

  if n_cont < (0.5e6 * len(chromos)) and not is_single_cell:
    util.warn('Contact map is sparse but single-cell "-sc" option not used')
    
  if contacts2:
    if file_bin_size2:
      ret = get_contact_arrays_matrix(contacts2, bin_size, chromos, chromo_limits, file_bin_size2, chromo_limits2)
      count_list2, full_matrix2, label_pos2, offsets2 = ret
      n_cont2, n_cis2, n_trans2, n_homolog2, n_ambig2 = count_list2
      ambig_matrix2 = None
      n_isol2 = None
      trans_counts2 = None
    else:
      ret = get_contact_lists_matrix(contacts2, bin_size, chromos, chromo_limits)
      count_list2, full_matrix2, ambig_matrix2, label_pos2, offsets2, trans_counts2, ambig_groups2 = ret
      n_cont2, n_cis2, n_trans2, n_homolog2, n_ambig2, n_pairs2, n_isol2 = count_list2
  
  data_track_dicts = {}

  for data_track_path, file_format, text_label in data_tracks:
    if file_format == GFF_FORMAT:
      data_dict = gff.load_data_track(data_track_path, gff_feats, merge=True)

    elif file_format == WIG_FORMAT:
      data_dict = wig.load_data_track(data_track_path)
    
    elif file_format == BED_FORMAT:
      data_dict = bed.load_data_track(data_track_path)
      
    else: # BAM/SAM
      data_dict = sam.load_data_track(data_track_path, SAM_BIN_SIZE)
           
    if set(data_dict) & set(chromos):
      data_track_dicts[text_label] = data_dict
    else:
      util.warn('{} file chromosome names do not correspond to any contact data chromosomes'.format(file_format))

  if use_corr:
    util.info('Calculating correlation matrix')
    has_neg = True
    full_matrix = get_corr_mat(full_matrix, chromo_offsets=offsets)

    if contacts2:
      full_matrix2 = get_corr_mat(full_matrix2, chromo_offsets=offsets2)

  else:
    has_neg = full_matrix.min() < 0

  if has_neg or is_single_cell:
    use_log = False
  else:
    use_log = True

  if use_corr:
    metric = 'Correlation'
    v_max = 0.3
    v_min = -0.3
 
  elif has_neg: # Some external non-count value being plotted
    metric = 'Value '
    v_max = None # Will be set to full, symmetric range
    v_min = None
 
  elif is_single_cell:
    metric = 'Count'
    v_max = 4
    v_min = 0
 
  elif use_norm: # Normalised log scale counts
    metric = 'Count/mean'
    norm_scale = np.mean(full_matrix[full_matrix.nonzero()])
    full_matrix = full_matrix.astype(float)/norm_scale

    v_max = _get_vmax(full_matrix)
    v_min = 1.0/norm_scale
 
  else: # Log counts
    metric = 'Count'
    v_max = _get_vmax(full_matrix)
    v_min = 1.0

  if cmap:
    colors = cmap
    bad_color = '#FFFFFF'
  
  elif black_bg:
    if has_neg:
      colors = ['00FFFF', '#0000FF', '#000000', '#FF0000', '#FFFF00']
    else:
      colors = ['#000000', '#BB0000', '#DD8000', '#FFFF00', '#FFFF80','#FFFFFF']
    
    bad_color = '#404040'

  else:
    if has_neg:
      colors = ['#0000B0', '#0080FF', '#FFFFFF', '#FF0000', '#800000']
      
    elif is_single_cell:
      colors = ['#FFFFFF', '#0080FF' ,'#0000FF','#0000B0','#000080']
    
    else:
      colors = ['#FFFFFF', '#0080FF' ,'#FF0000','#FFFF00','#000000']
      colors = ['#FFFFFF', '#0040FF' ,'#FF0000']
    
    bad_color = '#B0B0B0'

  if region_dict:
  
    for chr_a in region_dict:
      pair = (chr_a, chr_a)
      limits = chromo_limits[chr_a]
      
      for region in region_dict[chr_a]:
        data_tracks = []
        s, e = region
        
        if data_track_dicts:
          for ft in sorted(data_track_dicts):
            if chr_a in data_track_dicts[ft]:
              track = data_track_dicts[ft][chr_a]
              in_region = (track['pos1'] < e) & (track['pos1'] > s) | (track['pos2'] < e) & (track['pos2'] > s)
              track = track[in_region]
              data_tracks.append((ft, track))
              
        if file_bin_size:
          matrix = get_region_array_matrix(contacts[pair], limits, region,
                                           file_bin_size, bin_size2)
          ambig_matrix = None
 
        else:
          matrix, ambig_matrix = get_region_list_matrix(contacts[pair], region,
                                                        bin_size2, ambig_groups, smooth)
        
        cn_cont = int(matrix.sum()) // 2
 
        if use_corr:
          matrix = get_corr_mat(matrix)
          
          if contacts2:
            if file_bin_size:
              matrix2 = get_region_array_matrix(contacts2[pair], limits, region,
                                                file_bin_size, bin_size2)
            else:
              matrix2, ambig_matrix2 = get_region_list_matrix(contacts2[pair], region,
                                                              bin_size2, ambig_groups, smooth)
 
            cn_cont2 = int(matrix2.sum()) // 2
            matrix2 = get_corr_mat(matrix2)
            idx = np.tril_indices(len(matrix), 1)
            matrix[idx] = matrix2[idx]
 
          cv_max = v_max
          cv_min = v_min
 
        elif contacts2:
          if file_bin_size:
            matrix2 = get_region_array_matrix(contacts2[pair], limits, region,
                                              file_bin_size, bin_size2)
          else:
            matrix2, ambig_matrix2 = get_region_list_matrix(contacts2[pair], region,
                                                            bin_size2, ambig_groups, smooth)
 
          cn_cont2 = int(matrix2.sum()) // 2
          m = len(matrix)
          idx = np.tril_indices(m, 1)
          matrix[idx] = matrix2[idx]
 
        if use_corr or has_neg or is_single_cell: # Scale range same as global
          cv_max = v_max
          cv_min = v_min
 
        elif use_norm:
          cv_max = v_max
          cv_min = v_min
          nz = matrix[matrix > 0]
 
          if len(nz):
            norm_scale = np.mean(matrix[matrix.nonzero()])
            matrix = matrix.astype(float)/norm_scale

            cv_max = _get_vmax(matrix)
            cv_min = 1.0/norm_scale

        else:
          cv_max = _get_vmax(matrix)
          cv_min = 1.0
 
        title = 'Chromosome %s : %.3f-%.3f Mb' % (chr_a, s/1e6, e/1e6)

        dw = None
        bs2 = bin_size2/1e3
        scale_label = '%s (%.1f kb bins)' % (metric, bs2)
        if diag_width:
          dw = min(diag_width, e-s)
 
        if contacts2:
          stats_text = 'Contacts:{:,d}\{:,d}'.format(cn_cont2, cn_cont)
        else:
          stats_text = 'Contacts:{:,d}'.format(cn_cont)
 
        double_diag = contacts2 and diag_width
 
        plot_contact_matrix(matrix, bin_size2, title, scale_label, None, pair,
                            chromo_grid, stats_text, colors, bad_color,
                            x_data_tracks=data_tracks, y_data_tracks=data_tracks, log=use_log, pdf=pdf,
                            v_max=cv_max, v_min=cv_min, ambig_matrix=ambig_matrix, diag_width=dw,
                            double_diag=double_diag, x_start=s, y_start=s)
  
  else:  
    
    n = len(full_matrix)
    util.info('Full contact map size %d x %d' % (n, n))
 
    f_cis = 100.0 * n_cis / float(n_cont or 1)
    f_trans = 100.0 * n_trans / float(n_cont or 1)
 
    if contacts2:
      idx = np.tril_indices(n, 1)
      full_matrix[idx] = full_matrix2[idx]
      f_cis2 = 100.0 * n_cis2 / float(n_cont2 or 1)
      f_trans2 = 100.0 * n_trans2 / float(n_cont2 or 1)

    if has_neg and not use_corr:
      stats_text = ''
 
    else:
      if contacts2:
        stats_text = 'Contacts:{:,d}\{:,d} cis:{:,d}\{:,d} ({:.1f}%\{:.1f}%) ' \
                     'trans:{:,d}\{:,d} ({:.1f}%\{:.1f}%)'
        stats_text = stats_text.format(n_cont, n_cont2, n_cis, n_cis2,
                                       f_cis, f_cis2, n_trans, n_trans2,
                                       f_trans, f_trans2)
 
      else:
        stats_text = 'Contacts:{:,d} cis:{:,d} ({:.1f}%) trans:{:,d} ({:.1f}%)'
        stats_text = stats_text.format(n_cont, n_cis, f_cis, n_trans, f_trans)
 
      if n_homolog:
        f_homolog = 100.0 * n_homolog / float(n_cont or 1)
 
        if contacts2:
          f_homolog2 = 100.0 * n_homolog2 / float(n_cont2 or 1)
          extra = ' homolog:{:,d}\{:,d} ({:.1f}%\{:.1f}%)'
          stats_text += extra.format(n_homolog, n_homolog2, f_homolog, f_homolog2)

        else:
          extra = ' homolog:{:,d} ({:.1f}%)'
          stats_text += extra.format(n_homolog, f_homolog)
 
    
    if len(chromos) > 1:
      mito_frac, mito_cat = _get_mito_fraction(contacts, file_bin_size)
 
      if contacts2:
        mito_frac2, mito_cat2 = _get_mito_fraction(contacts2, file_bin_size2)
        
      if in_path2:
        title = os.path.basename(in_path2) + '\\' + os.path.basename(in_path)
      else:
        title = os.path.basename(in_path)
 
      grid = [offsets[c][1] for c in chromos[1:]]
      scale_label = '%s (%.2f Mb bins)' % (metric, bin_size/1e6)
      extra_texts = []
 
      if n_isol is not None:
        isol_frac = 100.0 * n_isol / float(n_pairs or 1)
 
        if contacts2:
          isol_frac2 = 100.0 * n_isol2 / float(n_pairs2 or 1)
          extra_texts.append('Isolated:{:.2f}%\{:.2f}%'.format(isol_frac, isol_frac2))
        else:
          extra_texts.append('Isolated:{:.2f}%'.format(isol_frac))
 
      if trans_counts:
        trans_dev, ploidy = _get_trans_dev(trans_counts)
 
        if contacts2:
          trans_dev2, ploidy2 = _get_trans_dev(trans_counts2)
          txt = 'Ploidy score:{:.2f}\{:.2f} ({}\{})'
          extra_texts.append(txt.format(trans_dev, trans_dev2, ploidy, ploidy2))
        else:
          txt = 'Ploidy score:{:.2f} ({})'
          extra_texts.append(txt.format(trans_dev, ploidy))
 
      if contacts2:
        txt = 'Mito score:{:.2f}\{:.2f} ({}\{})'
        extra_texts.append(txt.format(mito_frac, mito_frac2, mito_cat, mito_cat2))
      else:
        txt = 'Mito score:{:.2f} ({})'
        extra_texts.append(txt.format(mito_frac, mito_cat))
 
      stats_text += '\n' + ' ; '.join(extra_texts)
 
      plot_contact_matrix(full_matrix, bin_size, title, scale_label, zip(label_pos, chromo_labels),
                          None, grid, stats_text, colors, bad_color, log=use_log, pdf=pdf,
                          v_max=v_max, v_min=v_min, ambig_matrix=ambig_matrix)
 
    clim = None
    #clim = 650000
    #clim = 2698000
    #clim = 1632000
    
    if separate_cis or separate_trans:
      pairs = []
 
      if separate_cis:
        for chr_a in chromos:
          pair = (chr_a, chr_a)
 
          if pair in contacts:
            pairs.append(pair)
 
      if separate_trans:
        for i, chr_a in enumerate(chromos[:-1]):
          for chr_b in chromos[i+1:]:
            pair = (chr_a, chr_b)
 
            if pair in contacts:
              pairs.append(pair)
 
      for pair in pairs:
        chr_a, chr_b = pair # Display order
        is_cis = chr_a == chr_b
        key = tuple(sorted(pair)) # Key order
        limits_a = chromo_limits[key[0]]
        limits_b = chromo_limits[key[1]]
 
        if contacts2:
          limits_a2 = chromo_limits2[key[0]]
          limits_b2 = chromo_limits2[key[1]]
 
        pair_bin_size = bin_size2 if is_cis else bin_size3
 
        if file_bin_size:
          matrix = get_single_array_matrix(contacts[key], limits_a, limits_b, is_cis,
                                           file_bin_size, pair_bin_size)
          ambig_matrix = None
        else:
          matrix, ambig_matrix = get_single_list_matrix(contacts[key], limits_a, limits_b,
                                                        is_cis, pair_bin_size, ambig_groups, smooth)
        if clim:
          matrix = limit_counts(matrix, clim)
        
        if key != pair:
          matrix = matrix.T
 
        cn_cont = int(matrix.sum())
        if is_cis:
          cn_cont //= 2
 
        if use_corr:
          if is_cis:
            matrix = get_corr_mat(matrix)
 
            if contacts2:
              if file_bin_size:
                orig_lim = limits_a2 + limits_b2
                matrix2 = get_single_array_matrix(contacts2[key], limits_a2, limits_b2, is_cis,
                                                  file_bin_size, pair_bin_size, orig_lim)
              else:
                matrix2, ambig_matrix2 = get_single_list_matrix(contacts2[key], limits_a2, limits_b2,
                                                                is_cis, pair_bin_size, ambig_groups, smooth)
              
              if clim:
                matrix2 = limit_counts(matrix2, clim)
              
              cn_cont2 = int(matrix2.sum()) // 2
              matrix2 = get_corr_mat(matrix2)
              idx = np.tril_indices(len(matrix), 1)
              matrix[idx] = matrix2[idx]
 
          else:
            if file_bin_size:
              matrix_a = get_single_array_matrix(contacts[(chr_a, chr_a)], limits_a, limits_a,
                                                 True, file_bin_size, pair_bin_size)
            else:
              matrix_a, amb_a = get_single_list_matrix(contacts[(chr_a, chr_a)], limits_a, limits_a,
                                                       True, pair_bin_size, ambig_groups, smooth)

            if file_bin_size:
              matrix_b = get_single_array_matrix(contacts[(chr_b, chr_b)], limits_b, limits_b,
                                                 True, file_bin_size, pair_bin_size)
            else:
              matrix_b, amb_b = get_single_list_matrix(contacts[(chr_b, chr_b)], limits_b, limits_b,
                                                       True, pair_bin_size, ambig_groups, smooth)

            matrix = get_trans_corr_mat(matrix_a, matrix_b, matrix)
 
          cv_max = v_max
          cv_min = v_min
 
        elif contacts2 and is_cis:
          if file_bin_size:
            matrix2 = get_single_array_matrix(contacts2[key], limits_a, limits_b, is_cis,
                                              file_bin_size, pair_bin_size, limits_a2 + limits_b2)
          else:
            matrix2, ambig_matrix2 = get_single_list_matrix(contacts2[key], limits_a, limits_b, is_cis,
                                                            pair_bin_size, ambig_groups, smooth)
 
          cn_cont2 = int(matrix2.sum()) // 2
          m = len(matrix)
          idx = np.tril_indices(m, 1)
          matrix[idx] = matrix2[idx]
 
        if use_corr or has_neg or is_single_cell: # Scale range same as global
          cv_max = v_max
          cv_min = v_min
 
        elif use_norm:
          cv_max = v_max
          cv_min = v_min
          nz = matrix[matrix > 0]
 
          if len(nz):
            norm_scale = np.mean(matrix[matrix.nonzero()])
            matrix = matrix.astype(float)/norm_scale

            if is_cis:
              cv_max = _get_vmax(matrix)
            else:
              cv_max = matrix.max()
 
            cv_min = 1.0/norm_scale

        else:
          if is_cis:
            cv_max = _get_vmax(matrix)
          else:
            cv_max = matrix.max()
 
          cv_min = 1.0
 
        title = 'Chromosome %s' % chr_a if is_cis else 'Chromosomes %s - %s ' % pair
        dw = None
        
        pbs = pair_bin_size/1e3
        
        if is_cis:
          scale_label = '%s (%.1f kb bins)' % (metric, pbs)
          if diag_width:
            dw = diag_width
 
        else:
          scale_label = '%s (%.3f Mb bins)' % (metric, pbs/1e3)
 
        if contacts2 and is_cis:
          stats_text = 'Contacts:{:,d}\{:,d}'.format(cn_cont2, cn_cont)
        else:
          stats_text = 'Contacts:{:,d}'.format(cn_cont)
 
        double_diag = is_cis and contacts2 and diag_width

        x_data_tracks = []
        y_data_tracks = []
        
        if data_track_dicts:
          for ft in sorted(data_track_dicts):
            if chr_a in data_track_dicts[ft]:
              x_data_tracks.append((ft, data_track_dicts[ft][chr_a]))
            else:
              x_data_tracks.append((ft, np.array([], dtype=util.DATA_TRACK_TYPE)))
            
            if chr_b in data_track_dicts[ft]:
              y_data_tracks.append((ft, data_track_dicts[ft][chr_b]))
            else:
              y_data_tracks.append((ft, np.array([], dtype=util.DATA_TRACK_TYPE)))
              
        plot_contact_matrix(matrix, pair_bin_size, title, scale_label, None, pair,
                            chromo_grid, stats_text, colors, bad_color,
                            x_data_tracks=x_data_tracks, y_data_tracks=y_data_tracks,
                            log=use_log, pdf=pdf, v_max=cv_max, v_min=cv_min,
                            ambig_matrix=ambig_matrix, diag_width=dw, double_diag=double_diag)
                        
  if pdf:
    pdf.close()
    util.info('Written {}'.format(out_path))
  else:
    util.info('Done')
  
  
def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument(metavar='CONTACT_FILE', nargs='+', dest='i',
                         help='One or two input NPZ (binned, bulk Hi-C data) or NCC format (single-cell) chromatin contact file(s). Wildcards accepted. ' \
                              'If two files are input these will be displayed either side of the diagonal (does not apply for -t option).')

  arg_parse.add_argument('-o', metavar='OUT_FILE', default=None,
                         help='Optional output file name. If not specified, a default based on the input file name and output format will be used.')
  
  arg_parse.add_argument('-chr', metavar='CHROMOSOMES', nargs='+', default=None,
                         help='Optional selection of chromsome names to generate contact maps for. ' \
                              'Otherwise all chromosomes/contigs above the minimum size (see -m options) are used')

  arg_parse.add_argument('-bed', '--bed-data-track', metavar='BED_FILE', nargs='*', default=None, dest="bed",
                         help='Optional BED format (inc. broadPeak/narrowPeak) files for displaying ancilliary data along single-chromosome axes. Wildcards accepted. ' \
                              'Short data names may be supplied to avoid graphical clutter using the short_label@long_file_path format, e.g. H3K4me3@/data/GSM12345_H3K4me3_macs2_peaks.bed labels the data as "H3K4me3"')
  
  arg_parse.add_argument('-sam', '--sam-data-track', metavar='SAME_FILE', nargs='*', default=None, dest="sam",
                         help='Optional SAM or BAM format (inc. broadPeak/narrowPeak) files for displaying ancilliary data along single-chromosome axes. Wildcards accepted. ' \
                              'Short data names may be supplied to avoid graphical clutter using the short_label@long_file_path format.')
  
  arg_parse.add_argument('-wig', '--wig-data-track', metavar='WIG_FILE', nargs='*', default=None, dest="wig",
                         help='Optional Wiggle format (variable of fixed step) files for displaying ancilliary data along single-chromosome axes. Wildcards accepted. ' \
                              'Short data names may be supplied to avoid graphical clutter using the short_label@long_file_path format.')

  arg_parse.add_argument('-gff', '--gff-data-track', metavar='GENOME_FEATURE_FILE', nargs='*', default=None, dest="gff",
                         help='Optional genome feature files (GFF/GTF format) for displaying gene locations etc. along single-chromosome axes. Wildcards accepted. ' \
                              'Displays gene/exon/CDS features in the file unless the -gfff option is used. Short source names may be supplied to avoid graphical ' \
                              'clutter using the short_label@long_file_path format.')

  arg_parse.add_argument('-gfff', '-gff-feature', default=None, nargs='*', metavar='GFF_FEATURE_NAME', dest="gfff",
                         help='One or more feture types to display from input GFF/GTF format data track. Default: "gene exon CDS"')
  
  arg_parse.add_argument('-sc', '--single-cell', default=False, action='store_true', dest="sc",
                         help='Specifies that the input data is from single-cell Hi-C')

  arg_parse.add_argument('-nc', '--no-cis', default=False, action='store_true', dest="nc",
                         help='Do not display separate contact maps for individual chromosomes (intra-chromosomal contacts). ' \
                              'Only the overall whole-genome map will be displayed (unless -t option also used).')

  arg_parse.add_argument('-t', '--trans', default=False, action='store_true', dest="t",
                         help='Display separate contact maps for all trans (inter-chromosomal) pairs. ' \
                              'By default the overall whole-genome and intra-chromosome maps are generated.')

  arg_parse.add_argument('-r', '--regions', default=None, nargs='+', dest="r",
                         help='Display contact maps for only specified regions rather than whole chromsomes. ' \
                              'Regions should be specified as chromosome:start-end with positions in Mb, ' \
                              'e.g. "chr3:5.35-10.35". Overrides -chr, -t and -m options.')

  arg_parse.add_argument('-g', '--screen-gfx', default=False, action='store_true', dest='g',
                         help='Display graphics on-screen using matplotlib, where possible and do not automatically save output.')

  arg_parse.add_argument('-s1', '--bin-size-main', default=None, metavar='KB_BIN_SIZE', type=float, dest="s1",
                         help='Binned sequence region size (the resolution) for the overall, whole-genome contact map, in kilobases. ' \
                              'Default is adaptive to give at least 1000 bins.')

  arg_parse.add_argument('-s2', '--bin-size-cis', default=None, metavar='KB_BIN_SIZE', type=float, dest="s2",
                         help='Binned sequence region size (the resolution) for separate intra-chromsomal maps, ' \
                              'in kilobases. Default is adaptive to give at least 10 bins for the smallest chromosome.')
  
  arg_parse.add_argument('-s3', '--bin-size-trans', default=None, metavar='KB_BIN_SIZE', type=float, dest="s3",
                         help='Binned sequence region size (the resolution) for separate inter-chromsomal maps, ' \
                              'in kilobases. Default is adaptive based on the smallest allowed chromosome')

  arg_parse.add_argument('-m', default=0.0, metavar='MIN_CONTIG_SIZE', type=float,
                         help='The minimum chromosome/contig sequence length in Megabases for inclusion. ' \
                              'Default is {}%% of the largest chromosome/contig length.'.format(DEFAULT_SMALLEST_CONTIG*100))

  arg_parse.add_argument('-sm', '--smooth', default=False, action='store_true', dest='sm',
                         help='For NCC format contacts only, use fractional binning to get smoother chromosome maps. Does not apply to whole genome map.')

  arg_parse.add_argument('-bbg' '--black-bg',default=False, action='store_true', dest="bbg",
                         help='Specifies that the contact map should have a black background (default is white).')

  arg_parse.add_argument('-corr', default=False, action='store_true', dest="corr",
                         help='Plot Pearson correlation coefficients for the contacts, rather than counts. ' \
                              'For trans/inter-chromosome pairs, the correlations shown are the non-cis part of the '\
                              'square, symmetric correlation matrix of the combined map for both chromosomes.')

  arg_parse.add_argument('-norm', default=False, action='store_true', dest="norm",
                         help='Normalise the contact map so that displayed values are relative to (non-zero) mean of the contact map; ' \
                              'useful for comparing maps with different numbers of contacts. ' \
                              'This option will be ignored if the -corr option is specified.')

  arg_parse.add_argument('-grid', default=False, action='store_true', dest="grid",
                         help='Show grid lines at numeric chromosome positions.')

  arg_parse.add_argument('-diag', default=0.0, metavar='REGION_WIDTH', const=DEFAULT_DIAG_REGION, type=float, dest="diag", nargs='?',
                         help='Plot horizontally only the diagonal parts of the intra-chromosomal contact matrices. ' \
                              'The width of stacked regions (in Megabases) may be optionally specified, ' \
                              'but otherwise defaults to %.1f Mb' % DEFAULT_DIAG_REGION)
  
  arg_parse.add_argument('-colors', metavar='COLOR_MAP', default=None,
                         help='Optional comma-separated map colours, e.g. "white,blue,red".' \
                              'or colormap (scheme) name, as used by matplotlib. ' \
                              'Note: #RGB style hex colours must be quoted e.g. "#FF0000,#0000FF" ' \
                              'See: %s This option overrides -b.' % COLORMAP_URL)

  args = vars(arg_parse.parse_args(argv))

  in_paths = args['i']
  out_path = args['o']
  screen_gfx = args['g']
  bin_size = args['s1']
  bin_size2 = args['s2']
  bin_size3 = args['s3']
  min_contig_size = args['m']
  black_bg = args['bbg']
  no_sep_cis = args['nc']
  sep_trans = args['t']
  chromos = args['chr']
  use_corr = args['corr']
  use_norm = args['norm']
  is_single = args['sc']
  chromo_grid = args['grid']
  diag_width = args['diag']
  cmap = args['colors']
  regions = args['r']
  bed_paths = args['bed'] or []
  wig_paths = args['wig'] or []
  sam_paths = args['sam'] or []
  gff_paths = args['gff'] or []
  gff_feats = args['gfff']
  smooth = args['sm']
  
  if not in_paths:
    arg_parse.print_help()
    sys.exit(1)
  
  if len(in_paths) > 2:
    util.critical('Only one or two input datasets may be specified')
  
  if out_path and screen_gfx:
    util.warn('Output file will not be written in screen graphics (-g) mode')
    out_path = None
  
  if use_corr and use_norm:
    util.warn('Correlation plot option "-corr" cannot be used with normalization option "-norm". The latter will be ignored')
    use_norm = False
  
  if gff_feats and not gff_path:
    util.warn('GFF feature type has been specified but no input GFF file')
    gff_feats = None
  
  data_tracks = []
  check_paths = []
  for data_paths, file_format in ((bed_paths, BED_FORMAT), (wig_paths, WIG_FORMAT),
                                  (sam_paths, SAM_FORMAT), (gff_paths, GFF_FORMAT)):
    for data_path in data_paths:
      if '@' in data_path:
        p = data_path.rfind('@')
        text_label = data_path[:p]
        data_path = data_path[p+1:]
      else:
        text_label = None
      
      if file_format == GFF_FORMAT:
        if not gff_feats:
          from formats import gff
          feature_counts = gff.get_feature_count(data_path)
          fc = [(feature_counts[f], f) for f in feature_counts]
          fc.sort(reverse=True)
 
          if 'gene' in feature_counts:
            gff_feats = ('gene','exon','CDS')
          else:
            gff_feats = fc[0][1]
 
          util.info('GFF file feature selection defaulting to "{}"'.format(gff_feats[0]))
 
          fc = ['{}:{:,}'.format(f, c) for c, f in fc]
          util.info('Available GFF features in {}:\n   counts {}'.format(data_path, ' '.join(fc)))
     
      data_tracks.append((data_path, file_format, text_label))
      check_paths.append(data_path)
  
  check_paths += in_paths 
  for in_path in check_paths:
    io.check_invalid_file(in_path)
  
  if cmap:
    cmap = util.string_to_colormap(cmap)
  
  if regions:
    sep_trans = False
    chromos = None
    region_dict = defaultdict(list)
  
    for region in regions:
      match = REGION_PATT.match(region)
      
      if not match:
        util.critical('Chromosome region specification could not be interpreted. ' \
                      'Requires chromosome:start-end where start and end are numbers.')
      
      bsize = bin_size2 or DEFAULT_CIS_BIN_KB
      chromo, start, end = match.groups()
      start, end = sorted([float(start) * 1e6, float(end) * 1e6])
      nbins = int(ceil((end-start)/(1e3*bsize)))
      
      if nbins < MIN_REGION_BINS:
        msg = 'Chromosome region %s:%.f-%.f should be at least %d times ' \
              'the bin size (see -s2 option); currently %d times.'
        ratio = (MIN_REGION_BINS * bsize2)/(end-start)
        util.critical(msg % (chromo, start, end, MIN_REGION_BINS, nbins))
      
      region_dict[chromo].append((start, end))
  
  else:
    region_dict = {}
      
  contact_map(in_paths, out_path, bin_size, bin_size2, bin_size3,
              no_sep_cis, sep_trans, chromos, region_dict,
              use_corr, use_norm, is_single, screen_gfx, black_bg,
              min_contig_size, chromo_grid, diag_width, data_tracks,
              gff_feats, smooth, cmap=cmap)


if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()


"""
To-do
 * +/- strand kept together on tracks

Data tracks:
 * NPZ
 * VCF?
"""
  
