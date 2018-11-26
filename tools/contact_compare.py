import os, sys, math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict 
from scipy import sparse, stats

PROG_NAME = 'contact_compare'
VERSION = '1.0.0'
DESCRIPTION = 'Compare two Hi-C contact maps (NPZ format)'

def normalize_contacts(contact_dict, chromo_limits, bin_sizes, bin_size, compare_trans=False, clip=0.4):
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
    
    if (not compare_trans) and (not is_cis):
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
  
  
def contact_compare(in_path_a, in_path_b, out_path=None, pdf_path=None, compare_trans=False, min_contig_size=None): 
    
  from nuc_tools import util, io
  from formats import npz  
  from contact_map import  plot_contact_matrix
  
  if not out_path:
    out_path = io.merge_file_names(in_path_a, in_path_b)
  
  if not pdf_path:
    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
  
  file_bin_size_a, chromo_limits_a, contacts_a = npz.load_npz_contacts(in_path_a)
  file_bin_size_b, chromo_limits_b, contacts_b = npz.load_npz_contacts(in_path_b)

  if file_bin_size_a != file_bin_size_b:
    util.critical('Chromatin contact matrices to be compared must be beinned at the same resolution')

  if min_contig_size:
    min_contig_size = int(min_contig_size * 1e6)
  else:
    largest = max([e-s for s, e in chromo_limits_a.values()])
    min_contig_size = int(0.05*largest) 
    util.info('Min. contig size not specified, using 5% of largest: {:,} bp'.format(min_contig_size))
  
  bin_size = file_bin_size_a
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
        elif compare_trans:
          trans_pairs.append(key)  
  
  util.info('Normalisation')
  
  sizes = {}
  for key in cis_pairs:
    s1, e1 = chromo_limits_a[key[0]]
    s2, e2 = chromo_limits_b[key[0]]
    chromo_limits[key[0]] = (0, max(e1, e2))
    sizes[key[0]] = int(math.ceil(max(e1, e2)/file_bin_size_a))
  
  normalize_contacts(contacts_a, chromo_limits_a, sizes, bin_size, compare_trans=compare_trans)
  normalize_contacts(contacts_b, chromo_limits_b, sizes, bin_size, compare_trans=compare_trans)  
        
  util.info('Calulating differences')
  
  pdf = PdfPages(pdf_path)
  colors = ['#0000B0', '#0080FF', '#FFFFFF', '#FF0000', '#800000']
  watermark = 'nuc_tools.contact_compare'
  
  for key in cis_pairs:
    util.info(' .. {}'.format(key[0]), line_return=True)  
       
    obs_a = contacts_a[key].toarray()
    obs_b = contacts_b[key].toarray()
    
    n, m = obs_a.shape
        
    obs_a[obs_a < 1.0/n] = 0.0 # Avoid anything too small to avoid extremes
    obs_b[obs_b < 1.0/n] = 0.0          
    diff = np.zeros((n, n), np.float32)

    nz = (obs_a> 0) & (obs_b > 0)      
    vals_a = obs_a[nz]
    vals_b = obs_b[nz]
    
    """    
    corrs = []
        
    for i in range(n):
      row_a = obs_a[i]
      row_b = obs_b[i]
      
      idx = (row_a> 0) & (row_b > 0)
      row_a = row_a[idx]
      row_b = row_b[idx]
      
      if len(row_a):
        m_a = row_a.mean()
        m_b = row_b.mean()
        d_a = row_a-m_a
        d_b = row_b-m_b
        rho = (d_a * d_b).sum()
        
        if rho:
          rho /= np.sqrt((d_a*d_a).sum())
          rho /= np.sqrt((d_b*d_b).sum())  
        else:
          rho = -1

      else:
        rho = -1
      
      corrs.append(rho)  
    """     
    if len(vals_a) > 2:
      lr = stats.linregress(vals_a, vals_b)
      slope = lr[0]
      y0 = lr[1]
      obs_b[nz] -= y0
      obs_b[nz] /= slope


    for d in range(1, n):
      z_scores = np.zeros(n-d, np.float32)
      idx1 = np.array(range(n-d))
      idx2 = idx1 + d
      idx = (idx1, idx2)

      vals_a = obs_a[idx]
      vals_b = obs_b[idx]
      
      nz = (vals_a > 0) & (vals_b > 0)
      
      vals_a = vals_a[nz]
      vals_b = vals_b[nz]
               
      if len(vals_a) > 2:
        nz_deltas = vals_a - vals_b
        med = np.median(nz_deltas)
        std = 1.4826 * np.median(np.abs(nz_deltas-med))
        z = (nz_deltas - med)/(std or 1.0)
        z[z > 5.0] = 5.0
        z[z < -5.0] = -5.0
        z_scores[nz] = z

         
      diff[idx] = z_scores
      
    title = 'Chromosome %s' % key[0]
    scale_label = 'Z score (%.2f kb bins)' % (bin_size/1e3)
    
    plot_contact_matrix(diff+diff.T, bin_size, title, scale_label, chromo_labels=None, axis_chromos=key, grid=None,
                        stats_text=None, colors=colors, bad_color='#404040', log=False, pdf=pdf, watermark=watermark)
                        
    out_matrix[key] = sparse.csr_matrix(diff)
  
  util.info(' .. done {} chromosomes'.format(len(cis_pairs)), line_return=True)  
  
  if compare_trans:
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
      
      title = 'Chromosomes %s - %s ' % key
      scale_label = 'Z score (%.2f kb bins)' % (bin_size/1e3)
      
      plot_contact_matrix(z_scores, bin_size, title, scale_label, chromo_labels=None, axis_chromos=key, grid=None,
                          stats_text=None, colors=colors, bad_color='#404040', log=False, pdf=pdf, watermark=watermark)
 
      out_matrix[key] = sparse.coo_matrix(z_scores)

    util.info(' .. done {} pairs'.format(len(trans_pairs)), line_return=True)  
  
  if pdf:
    pdf.close()
    util.info('Written {}'.format(pdf_path))
 
  util.info('Saving data')
    
  npz.save_contacts(out_path, out_matrix, chromo_limits, file_bin_size_a, min_bins=0)  
  util.info('Written {}'.format(out_path))
  

# Test fit of differences to normality
# - could plot quantiles rather than Z-score

# Create optional PDF in this tool
# - borrow genralised contact_map functionality


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
                         help='Optional output NPZ format file name. If not specified, a default based on the input file names will be used.')

  arg_parse.add_argument('-p', metavar='OUT_PDF_FILE', default=None,
                         help='Optional PDF file to save report. If not specified, a default based on the input file names will be used..')

  arg_parse.add_argument('-m', default=0.0, metavar='MIN_CONTIG_SIZE', type=float,
                         help='The minimum chromosome/contig sequence length in Megabases for inclusion. ' \
                              'Default is 10% of the largest chromosome/contig length.')

  arg_parse.add_argument('-t', default=False, action='store_true',
                         help='Compare trans (inter-chromosomal) chromosome pairs. ' \
                              'By default only the intra-chromosomal contacts are compares')

  args = vars(arg_parse.parse_args(argv))

  in_path_a, in_path_b = args['i']
  out_path = args['o']
  pdf_path = args['p']
  comp_trans = args['t']
  min_contig_size = args['m']
  
  contact_compare(in_path_a, in_path_b, out_path, pdf_path, comp_trans, min_contig_size)
  
  
  
if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()
