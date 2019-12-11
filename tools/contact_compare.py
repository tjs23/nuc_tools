import os, sys, math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict 
from scipy import sparse, stats

PROG_NAME = 'contact_compare'
VERSION = '1.0.1'
DESCRIPTION = 'Compare two Hi-C contact maps (NPZ format)'
DEFAULT_SMALLEST_CONTIG = 0.1
DEFAULT_DMAX = 5.0
DEFAULT_CMAX = 0.5
DEFAULT_DIAG_REGION = 50.0

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
    
    s2, e2 = new_chromo_limits[chr_a] # Range in new data
    num_bins = int(math.ceil(e2/bin_size)) 
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
      
      n = min(off_a+a, len(contact_scale[chr_a])) # Matrix can exceed the new chromo limits
      m = min(off_b+b, len(contact_scale[chr_b]))
      
      contact_scale[chr_a][off_a:n] += orig_mat[:n-off_a].sum(axis=1)
      contact_scale[chr_b][off_b:m] += orig_mat[:,:m-off_b].sum(axis=0)
  
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
    
    #nz = scale.nonzero()[0]
    #scale *= len(nz)/scale[nz].sum()
    
    contact_scale[chr_a] = scale
  
  for chr_a, chr_b in pairs: # Sorted and available
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
    
    after_a = max(0, (lim_a-a-off_a))
    after_b = max(0, (lim_b-b-off_b))
    
    if off_a or off_b or after_a or after_b:
      # all pairs use full range from zero
      mat = np.pad(mat, [(off_a,after_a), (off_b,after_b)], 'constant') # will ensure square cis (it needn't be when only storing upper matrix)
      a, b = mat.shape
    
    if a > lim_a:
      a = lim_a
      mat = mat[:a]

    if b > lim_b:
      b = lim_b
      mat = mat[:,:b]
    
    if is_cis:
      mat -= np.diag(np.diag(mat))
      cols = np.arange(a-1)
      rows = cols-1
      
      sub_diag_a = mat[rows, cols]
      sub_diag_b = mat[cols, rows]
      
      # check for symmetry
      
      if np.all(sub_diag_a == sub_diag_b):
        contact_scale[chr_a] *= 2 # Everything was counted twice : divide by double the amount
      else:
        mat += mat.T
     
    scale_a = contact_scale[chr_a].astype(np.float32)
    scale_b = contact_scale[chr_b].astype(np.float32)
        
    mat *= np.sqrt(np.outer(scale_a, scale_b))
    
    nnz = len(scale_a.nonzero()[0]) * len(scale_b.nonzero()[0])
    
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

  
def contact_compare(in_path_a, in_path_b, pdf_path=None, npz_path=None, bin_size=None,
                    compare_trans=False, min_contig_size=None, d_max=None,
                    use_corr=False, bed_path=None, diag_width=None, screen_gfx=False): 
    
  from nuc_tools import util, io
  from formats import npz, bed 
  from contact_map import  plot_contact_matrix, get_corr_mat, get_trans_corr_mat
  
  if not d_max:
    if use_corr:
      d_max = DEFAULT_CMAX
    else:
      d_max = DEFAULT_DMAX
  
  if not pdf_path:
    if use_corr:
      suffix = '_corr'
    else:
      suffix = '_diff'
    
    pdf_path = io.merge_file_names(in_path_a, in_path_b, suffix=suffix)
    
  pdf_path = io.check_file_ext(pdf_path, '.pdf')
  
  if npz_path:
    npz_path = io.check_file_ext(npz_path, '.npz')

  file_bin_size_a, chromo_limits_a, contacts_a = npz.load_npz_contacts(in_path_a)
  file_bin_size_b, chromo_limits_b, contacts_b = npz.load_npz_contacts(in_path_b)

  if file_bin_size_a != file_bin_size_b:
    util.critical('Chromatin contact matrices to be compared must be binned at the same resolution')
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
    
  if screen_gfx:
    util.info('Displaying comparison map for {} vs {}'.format(in_path_a, in_path_b))
  else:
    util.info('Making PDF comparison map for {} vs {}'.format(in_path_a, in_path_b))
     
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
  
  if use_corr:
    util.info('Selected option to compare Pearson correlation coefficients')

  util.info('Normalisation')
  
  for key in cis_pairs:
    s1, e1 = chromo_limits_a[key[0]]
    s2, e2 = chromo_limits_b[key[0]]
    chromo_limits[key[0]] = (0, max(e1, e2))
  
  # Vanilla normalisation for now. Enforces comparable matrix sizes.
  normalize_contacts(contacts_a, chromo_limits_a, orig_bin_size, chromo_limits, bin_size, compare_trans=compare_trans)
  normalize_contacts(contacts_b, chromo_limits_b, orig_bin_size, chromo_limits, bin_size, compare_trans=compare_trans)  
    
  if bed_path:
    bed_region_dict = {}
    bed_value_dict = {}
          
  util.info('Calulating differences')
  
  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(pdf_path)
  
  colors = ['#0000B0', '#0080FF', '#BBDDFF', '#FFFFFF', '#FFBBBB', '#FF0000', '#800000']
  watermark = 'nuc_tools.contact_compare'
  name_a = os.path.splitext(os.path.basename(in_path_a))[0]
  name_b = os.path.splitext(os.path.basename(in_path_b))[0]
  
  stats_text = 'Comparing %s to %s' % (name_a, name_b)
  legend = [] # ('-ve to +ve', colors[-2]), ('+ve to -ve', colors[1])]  
  bed_max = 0
  
  for key in cis_pairs:
    util.info(' .. comparing {}'.format(key[0]), line_return=True)  
      
    obs_a = contacts_a[key]
    obs_b = contacts_b[key]
    
    if hasattr(obs_a, 'toarray'):
      obs_a = obs_a.toarray()
      
    if hasattr(obs_b, 'toarray'):
      obs_b = obs_b.toarray()
      
    obs_a = obs_a.astype(float)
    obs_b = obs_b.astype(float)
    
    n, m = obs_a.shape        
    diff = np.zeros((n, n), np.float32)

    nz = (obs_a > 0) & (obs_b > 0)      
    vals_a = obs_a[nz]
    vals_b = obs_b[nz]
    
    if not len(vals_a):
      continue

    if not len(vals_b):
      continue
    
    if len(vals_a) > 2:
      slope, intercept, r_value, p_value, std_err = stats.linregress(vals_a, vals_b)
      obs_b[nz] -= intercept
      obs_b[nz] /= slope
      r2 = r_value*r_value
    else:
      r2 = 0.0
    
    if use_corr:
      nz = (obs_a != 0.0) & (obs_b != 0.0)
      x = get_corr_mat(obs_a)
      y = get_corr_mat(obs_b)
      
      diff = y - x
      diff[nz] = 0
      
      bed_values = np.abs(diff).sum(axis=0)

      scale_label = 'Correlation change (%.2f kb bins)' % (bin_size/1e3)

    else:
    
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
          deltas[nz] = vals_a - vals_b
 
        diff[idx] = deltas
      
      diff += diff.T  
      
      bed_values = diff.sum(axis=0)
        
      scale_label = 'Scaled difference (%.2f kb bins)' % (bin_size/1e3)

    title = 'Chromosome %s ; R2 = %.3f' % (key[0], r2)
    
    plot_contact_matrix(diff, bin_size, title, scale_label, chromo_labels=None, axis_chromos=key,
                        grid=None, stats_text=stats_text, colors=colors, bad_color='#404040', log=False,
                        pdf=pdf, watermark=watermark, legend=legend, v_max=d_max, diag_width=diag_width)
                        
    out_matrix[key] = sparse.csr_matrix(diff)
    
    if bed_path:
      pos = np.arange(0, bin_size*n, bin_size)
      bed_regions = np.stack([pos, pos+(bin_size-1)], axis=1)

      nz = bed_values.nonzero()
      bed_regions = bed_regions[nz]
      bed_values = bed_values[nz]
      
      if len(bed_values):
        bed_region_dict[key[0]] = bed_regions
        bed_value_dict[key[0]] = bed_values
        bed_max = max(bed_max, bed_values.max(), abs(bed_values.min()))
  
  util.info(' .. done {} chromosomes'.format(len(cis_pairs)), line_return=False)  
  
  if compare_trans:
    for key in trans_pairs:
      util.info(' .. comparing {} - {}'.format(*key), line_return=True)
      
      obs_a = contacts_a[key].toarray()
      obs_b = contacts_b[key].toarray()
      
      if use_corr:
        chr1, chr2 = key
        cis_a1 = contacts_a[(chr1, chr1)].toarray()
        cis_a2 = contacts_a[(chr2, chr2)].toarray()
        cis_b1 = contacts_b[(chr1, chr1)].toarray()
        cis_b2 = contacts_b[(chr2, chr2)].toarray()
      
        x = get_trans_corr_mat(cis_a1, cis_a2, obs_a)
        y = get_trans_corr_mat(cis_b1, cis_b2, obs_b)
 
        diff = x - y
        
        scale_label = 'Correlation change (%.2f kb bins)' % (bin_size/1e3)

      else:
 
        n, m = obs_a.shape
 
        nz = (obs_a > 0) & (obs_b > 0)
 
        vals_a = obs_a[nz]
        vals_b = obs_b[nz]
 
        vals_a /= np.median(vals_a)
        vals_b /= np.median(vals_b)
 
        diff = np.zeros((n, m), np.float32)
 
        if len(vals_a) > 2:
          diff[nz] = vals_a - vals_b
          
        scale_label = 'Scaled difference (%.2f kb bins)' % (bin_size/1e3)
 
      
      title = 'Chromosomes %s - %s ' % key
      
      plot_contact_matrix(diff, bin_size, title, scale_label, chromo_labels=None,
                          axis_chromos=key, grid=None, stats_text=None, colors=colors,
                          bad_color='#404040', log=False, pdf=pdf, watermark=watermark, 
                          legend=legend, v_max=d_max)
 
      out_matrix[key] = sparse.coo_matrix(diff)

    util.info(' .. done {} pairs'.format(len(trans_pairs)), line_return=True)  
  
  if bed_path:
    for chomo in bed_value_dict:
      bed_value_dict[chomo] = (1000.0/bed_max * bed_value_dict[chomo]).astype(int)
  
    bed.save_bed_data_track(bed_path, bed_region_dict, bed_value_dict)
    util.info('Written BED file {}'.format(bed_path))
  
  if pdf:
    pdf.close()
    util.info('Written PDF file {}'.format(pdf_path))
 
  util.info('Saving data')
  
  if npz_path:  
    npz.save_contacts(npz_path, out_matrix, chromo_limits, bin_size, min_bins=0)  
    util.info('Written NPZ file {}'.format(out_path))


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

  arg_parse.add_argument('-o', metavar='OUT_PDF_FILE', default=None,
                         help='Optional PDF file to save report. If not specified, a default based on the input file names will be used.')

  arg_parse.add_argument('-npz', metavar='OUT_NPZ_FILE', default=None,
                         help='Optional output NPZ format file name. If not specified, a default based on the input file names will be used.')

  arg_parse.add_argument('-s', '--bin-size', default=None, metavar='BIN_SIZE', type=float, dest="s",
                         help='Binned region size (the resolution) to compare contacts at, in kilobases. ' \
                              'Must be no smaller than the innate resolution of the inputs files. ' \
                              'Default is the innate resolution of the input files.')
  
  arg_parse.add_argument('-dmax', default=None, metavar='MAX_DISPLAY_DIFF', type=float,
                        help='The maximum +/- difference value for colour display; differences outside this value will be clipped. ' \
                             'Differences are either relative to sequence-separation median or between correlation coefficients (with -corr option). ' \
                             'Default is {:.1f} for count differences or {:.1f} for correlation differences.'.format(DEFAULT_DMAX, DEFAULT_CMAX))

  arg_parse.add_argument('-m', default=0.0, metavar='MIN_CONTIG_SIZE', type=float,
                        help='The minimum chromosome/contig sequence length in Megabases for inclusion. ' \
                              'Default is {}%% of the largest chromosome/contig length.'.format(DEFAULT_SMALLEST_CONTIG*100))

  arg_parse.add_argument('-t', default=False, action='store_true',
                         help='Compare trans (inter-chromosomal) chromosome pairs. ' \
                              'By default only the intra-chromosomal contacts are compared.')

  arg_parse.add_argument('-g', default=False, action='store_true',
                         help='Display graphics on-screen using matplotlib, where possible and do not automatically save output.')

  arg_parse.add_argument('-corr', default=False, action='store_true',
                         help='Compare differences in correlation matrices, rather than difference in sequence-separation normalized contact counts.' \
                              'For trans/inter-chromosome pairs, the correlations are taken from the non-cis part of the '\
                              'square, symmetric correlation matrix of the combined map for both chromosomes.')

  arg_parse.add_argument('-diag', default=0.0, metavar='REGION_WIDTH', const=DEFAULT_DIAG_REGION, type=float, dest="diag", nargs='?',
                         help='Plot horizontally only the diagonal parts of the intra-chromosomal contact matrices. ' \
                              'The width of stacked regions (in Megabases) may be optionally specified, ' \
                              'but otherwise defaults to %.1f Mb' % DEFAULT_DIAG_REGION)

  arg_parse.add_argument('-bed', metavar='OUT_BED_FILE', default=None,
                         help='Save differences (summed for each chromosome position) as a BED format file.')

  args = vars(arg_parse.parse_args(argv))

  in_path_a, in_path_b = args['i']
  npz_path = args['npz']
  pdf_path = args['o']
  bin_size = args['s']
  comp_trans = args['t']
  min_contig_size = args['m']
  screen_gfx = args['g']
  d_max = args['dmax']
  use_corr = args['corr']
  bed_path = args['bed']
  diag_width = args['diag']
  
  if not d_max:
    if use_corr:
      d_max = DEFAULT_CMAX
    else:
      d_max = DEFAULT_DMAX
  
  invalid_msg = io.check_invalid_file(in_path_a)
  if invalid_msg:
    util.critical(invalid_msg)

  invalid_msg = io.check_invalid_file(in_path_b)
  if invalid_msg:
    util.critical(invalid_msg)
  
  if io.is_same_file(in_path_a, in_path_b):
    util.warn('Inputs being compared are the same file')  
  
  if pdf_path and screen_gfx:
    util.warn('Output PDF file will not be written in screen graphics (-g) mode')
    pdf_path = None

  contact_compare(in_path_a, in_path_b, pdf_path, npz_path, bin_size,
                  comp_trans, min_contig_size, d_max, use_corr, bed_path,
                  diag_width, screen_gfx)
  

if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()
