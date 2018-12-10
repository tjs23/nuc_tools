import os, sys, math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict 
from scipy import sparse, stats

PROG_NAME = 'contact_compare'
VERSION = '1.0.0'
DESCRIPTION = 'Compare two Hi-C contact maps (NPZ format)'
DEFAULT_SMALLEST_CONTIG = 0.1
DEFAULT_DMAX = 5.0

import warnings
warnings.filterwarnings("ignore")

def normalize_contacts(contact_dict, chromo_limits, bin_size, new_chromo_limits=None,
                       new_bin_size=None, compare_trans=False, clip=0.4, store_sparse=True):
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
        
    scale_a = contact_scale[chr_a]
    scale_b = contact_scale[chr_b]
    
    mat *= np.sqrt(np.outer(scale_a, scale_b))

    nnz = np.sqrt(len(scale_a.nonzero()[0]) * len(scale_b.nonzero()[0]))
    
    mat *= nnz/(mat.sum() or 1.0) # The counts scale with the chromosome sizes
    
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
    

def _obs_vs_exp(obs_a, obs_b, n, clip=0.4):
  """
  Function not used but may be reinstated in the future
  """
  
  sep_dict_a = defaultdict(list)
  sep_dict_b = defaultdict(list)
  
  def _get_expectation(sig_seps, n):
    expt = np.zeros((n, n), float)
    for i in range(n):
      expt[i,:i] = sig_seps[:i][::-1]
      expt[i,i:] = sig_seps[:n-i]
 
    return expt
 
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

  comp = np.zeros((n, n), float)
  comp[nz] = 0.5 * (obs_a[nz] + obs_b[nz]) *  np.log(obs_a[nz]/obs_b[nz])

  return comp
  
  
def contact_compare(in_path_a, in_path_b, out_path=None, pdf_path=None, bin_size=None,
                    compare_trans=False, min_contig_size=None, d_max=DEFAULT_DMAX): 
    
  from nuc_tools import util, io
  from formats import npz  
  from contact_map import  plot_contact_matrix
  
  if not out_path:
    out_path = io.merge_file_names(in_path_a, in_path_b)
  
  if not pdf_path:
    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
  
  if not out_path.endswith('.npz'):
    out_path = out_path + '.npz'
  
  file_bin_size_a, chromo_limits_a, contacts_a = npz.load_npz_contacts(in_path_a)
  file_bin_size_b, chromo_limits_b, contacts_b = npz.load_npz_contacts(in_path_b)

  if file_bin_size_a != file_bin_size_b:
    util.critical('Chromatin contact matrices to be compared must be beinned at the same resolution')
    # Above could be relaxed as long as one is a multiple of the other, and lowest resolution is used

  if min_contig_size:
    min_contig_size = int(min_contig_size * 1e6)
  else:
    largest = max([e-s for s, e in chromo_limits_a.values()])
    min_contig_size = int(DEFAULT_SMALLEST_CONTIG*largest)
    msg = 'Min. contig size not specified, using {}% of largest: {:,} bp'
    util.info(msg.format(DEFAULT_SMALLEST_CONTIG*100, min_contig_size))
  
  orig_bin_size = file_bin_size_a
  
  if bin_size:
    bin_size *= 1e3
    
    if bin_size < orig_bin_size:
      msg = 'Comparison bin size (%.1f kb) cannot be smaller than the innate bin size in the input files (%.1f kb)'
      util.critical(msg % (bin_size/1e3, orig_bin_size/1e3))
    
  else:
    bin_size = orig_bin_size  
  
  # get a sorted list of large contigs/chromosomes common to both inputs
  
  common_keys = set(chromo_limits_a.keys()) & set(chromo_limits_b.keys())
  
  chromos = []
  for chromo in common_keys:
    
    if (chromo, chromo) not in contacts_a:
      continue
   
    if (chromo, chromo) not in contacts_b:
      continue
  
    s, e = chromo_limits_a[chromo]

    if (e-s) < min_contig_size:
      continue
    
    chromos.append(chromo)
    
  chromos = util.sort_chromosomes(chromos)

  if not chromos:
    util.critical('No sufficiently large chromosomes are common to both datasets')
  
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
  
  for key in cis_pairs:
    s1, e1 = chromo_limits_a[key[0]]
    s2, e2 = chromo_limits_b[key[0]]
    chromo_limits[key[0]] = (0, max(e1, e2))
  
  # Vanilla normalisation for now. Enforces comparable matrix sizes.
  normalize_contacts(contacts_a, chromo_limits_a, orig_bin_size, chromo_limits, bin_size, compare_trans=compare_trans)
  normalize_contacts(contacts_b, chromo_limits_b, orig_bin_size, chromo_limits, bin_size, compare_trans=compare_trans)  
        
  util.info('Calulating differences')
  
  pdf = PdfPages(pdf_path)
  colors = ['#0000B0', '#0080FF', '#BBDDFF', '#FFFFFF', '#FFBBBB', '#FF0000', '#800000']
  watermark = 'nuc_tools.contact_compare'
  name_a = os.path.splitext(os.path.basename(in_path_a))[0]
  name_b = os.path.splitext(os.path.basename(in_path_b))[0]
  legend = [(name_a, colors[-2]), (name_b, colors[1])]

  for key in cis_pairs:
    util.info(' .. comparing {}'.format(key[0]), line_return=True)  
            
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
    # May calculate overall correlation in the future...    
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
      slope, intercept, r_value, p_value, std_err = stats.linregress(vals_a, vals_b)
      obs_b[nz] -= intercept
      obs_b[nz] /= slope
      r2 = r_value*r_value
    else:
      r2 = 0.0

    for d in range(1, n):
      deltas = np.zeros(n-d, np.float32)
      idx1 = np.array(range(n-d))
      idx2 = idx1 + d
      idx = (idx1, idx2)

      vals_a = obs_a[idx]
      vals_b = obs_b[idx]
      
      nz = (vals_a > 0) & (vals_b > 0)
      
      vals_a = vals_a[nz]
      vals_b = vals_b[nz]
      
      vals_a /= np.median(vals_a)
      vals_b /= np.median(vals_b)
     
               
      if len(vals_a) > 2:
        nz_deltas = vals_a - vals_b
        nz_deltas[nz_deltas > d_max] = d_max
        nz_deltas[nz_deltas < -d_max] = -d_max
        deltas[nz] = nz_deltas
        
      diff[idx] = deltas
      
    title = 'Chromosome %s ; R2 = %.3f' % (key[0], r2)
    scale_label = 'Scaled difference (%.2f kb bins)' % (bin_size/1e3)
    
    plot_contact_matrix(diff+diff.T, bin_size, title, scale_label, chromo_labels=None, axis_chromos=key,
                        grid=None, stats_text=None, colors=colors, bad_color='#404040', log=False,
                        pdf=pdf, watermark=watermark, legend=legend, v_max=None)
                        
    out_matrix[key] = sparse.csr_matrix(diff)
  
  util.info(' .. done {} chromosomes'.format(len(cis_pairs)), line_return=True)  
  
  if compare_trans:
    for key in trans_pairs:
      util.info(' .. comparing {} - {}'.format(*key), line_return=True)
 
      vals_a = contacts_a[key].toarray()
      obs_b = contacts_b[key].toarray()
 
      n, m = vals_a.shape
 
      vals_a[vals_a < 1.0/n] = 0.0
      vals_b[vals_b < 1.0/n] = 0.0
      
      nz = (vals_a > 0) & (vals_b > 0)
      
      vals_a = vals_a[nz]
      vals_b = vals_b[nz]
      
      vals_a /= np.median(vals_a)
      vals_b /= np.median(vals_b)
 
      deltas = np.zeros((n, m), np.float32)
 
      if len(vals_a) > 2:
        deltas[nz] = vals_a - vals_b
        deltas[deltas > d_max] = d_max
        deltas[deltas < -d_max] = -d_max
      
      title = 'Chromosomes %s - %s ' % key
      scale_label = 'Scaled difference (%.2f kb bins)' % (bin_size/1e3)
      
      plot_contact_matrix(z_scores, bin_size, title, scale_label, chromo_labels=None,
                          axis_chromos=key, grid=None, stats_text=None, colors=colors,
                          bad_color='#404040', log=False, pdf=pdf, watermark=watermark, 
                          legend=legend, v_max=d_max)
 
      out_matrix[key] = sparse.coo_matrix(z_scores)

    util.info(' .. done {} pairs'.format(len(trans_pairs)), line_return=True)  
  
  if pdf:
    pdf.close()
    util.info('Written {}'.format(pdf_path))
 
  util.info('Saving data')
    
  npz.save_contacts(out_path, out_matrix, chromo_limits, bin_size, min_bins=0)  
  util.info('Written {}'.format(out_path))


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
                         help='Optional PDF file to save report. If not specified, a default based on the input file names will be used.')

  arg_parse.add_argument('-s', '--bin-size', default=None, metavar='BIN_SIZE', type=float, dest="s",
                         help='Binned region size (the resolution) to compare contacts at, in kilobases. ' \
                              'Must be no smaller than the innate resolution of the inputs files. ' \
                              'Default is the innate resolution of the input files.')
  
  arg_parse.add_argument('-dmax', default=DEFAULT_DMAX, metavar='MAX_DISPLAY_DIFF', type=float,
                        help='The maximum +/- difference value (scaled relative to sequence-separation median) for colour display. ' \
                             'Differences outside this value will be clipped. Default is {:.1f}.'.format(DEFAULT_DMAX))

  arg_parse.add_argument('-m', default=0.0, metavar='MIN_CONTIG_SIZE', type=float,
                        help='The minimum chromosome/contig sequence length in Megabases for inclusion. ' \
                              'Default is {}%% of the largest chromosome/contig length.'.format(DEFAULT_SMALLEST_CONTIG*100))

  arg_parse.add_argument('-t', default=False, action='store_true',
                         help='Compare trans (inter-chromosomal) chromosome pairs. ' \
                              'By default only the intra-chromosomal contacts are compared.')

  args = vars(arg_parse.parse_args(argv))

  in_path_a, in_path_b = args['i']
  out_path = args['o']
  pdf_path = args['p']
  bin_size = args['s']
  comp_trans = args['t']
  min_contig_size = args['m']
  d_max = args['dmax']

  invalid_msg = io.check_invalid_file(in_path_a)
  if invalid_msg:
    util.critical(invalid_msg)

  invalid_msg = io.check_invalid_file(in_path_b)
  if invalid_msg:
    util.critical(invalid_msg)
  
  if  io.is_same_file(in_path_a, in_path_b):
    util.warn('Inputs being compared are the same file')  
    
  contact_compare(in_path_a, in_path_b, out_path, pdf_path, bin_size,
                  comp_trans, min_contig_size, d_max)
  

if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()
