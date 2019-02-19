import sys
import os
import numpy as np
from matplotlib import pyplot as plt

ROG_NAME = 'contact_probability'
VERSION = '1.0.0'
DESCRIPTION = 'Chromatin contact (NCC or NPZ format) probability vs sequence separation and region analysis'
DEFAULT_BIN_SIZE = 100


def contact_probability(in_paths, pdf_path, bin_size=DEFAULT_BIN_SIZE, labels=None):

  from nuc_tools import util, io
  from formats import ncc, npz
  
  bin_size *= 1e3

  f, ax = plt.subplots()

  ax.set_alpha(0.5)
  ax.set_title('Contact sequence separations')

  x_limit = 10.0 ** 7.7 # log_max

  multi_set = len(in_paths) > 1

  for g, in_group in enumerate(in_paths):
    chromo_limits = {}
    contacts = {}
    seq_seps_all = []
    seq_seps = {}
    weights_all = []
    weights = {}

    n_files = len(in_group)

    for n, in_path in enumerate(in_group):
      seq_seps[in_path] = []
      weights[in_path] = []
      
      if in_path.lower().endswith('.ncc') or in_path.lower().endswith('.ncc.gz'):
        file_bin_size = None
        chromosomes, chromo_limits, contacts = ncc.load_file(in_path)
 
        for chromo_pair in contacts:
          chr_a, chr_b = chromo_pair
 
          if chr_a != chr_b:
            continue
 
          contact_array = np.array(contacts[chromo], np.int32)

          seps = abs(contact_array[:,0]-contact_array[:,1])

          indices = seps.nonzero()
          seps = seps[indices]

          p_start, p_end = chromo_limits[chromo]
          size = float(p_end-p_start+1)

          prob = (size/(size-seps)).tolist() # From fraction of chromosome that could give rise to each separation
          seps = seps.tolist()

          if not multi_set:
            seq_seps[in_path] += seps
            weights[in_path] += prob

          seq_seps_all += seps
          weights_all += prob
 
      else:
        file_bin_size, chromo_limits, contacts = npz.load_npz_contacts(in_path)
        
        # Go through all distances from the diagonal
        # - weights come from fraction of chromo and sum of counts
        
        
 
    seq_seps_all = np.array(seq_seps_all)

    num_bins = (x_limit-bin_size)/bin_size
    bins = np.linspace(bin_size, x_limit, num_bins)

    comb_y_data = None

    if not multi_set and (n_files > 1):
      for i, in_path in enumerate(seq_seps.keys()):
        data = np.array(seq_seps[in_path])

        hist, edges = np.histogram(data, bins=bins, weights=weights[in_path], normed=True)

        if comb_y_data is None:
          comb_y_data = np.zeros((n_files,len(hist)))

        idx = hist.nonzero()
        hist = hist[idx]
        comb_y_data[i,idx] = hist

    hist, edges = np.histogram(seq_seps_all, bins=bins, weights=weights_all, normed=True)
    idx = hist.nonzero()

    hist = hist[idx]
    edges = edges[idx]

    x_data = np.log10(edges)
    y_data = np.log10(hist)

    if not multi_set:
      y_err = comb_y_data.std(axis=0, ddof=1)[idx]
      y_lower = y_data - np.log10(hist-y_err)
      y_upper = np.log10(hist+y_err) - y_data

    y_min = int(2.0 * y_data.min())/2.0
    y_max = int(1.0 + 2.0 * y_data.max())/2.0

    if labels:
      label = labels.pop(0).replace('_', ' ')

    elif n_files > 1:
      if len(in_paths) > 1:
        label = 'Group %d' % (g+1)
      else:
        label = 'Combined datasets'

    else:
      label = None

    if multi_set:
      ax.plot(x_data, y_data, label=label, color=COLORS[g], linewidth=1, alpha=0.5)

    else:
      ax.fill_between(x_data, y_data-y_lower, y_data+y_upper, color='#FF0000', alpha=0.5, linewidth=0.5)
      ax.plot(x_data, y_data, label=label, color='#000000', alpha=1.0)
      ax.plot([],[], linewidth=8, color='#FF0000', alpha=0.5, label='$\pm\sigma$ over datasets')

  x_range = np.arange(np.log10(bin_size), np.log10(x_limit), 0.5)

  ax.set_xlabel('Sequence separation (bp)')
  ax.set_ylabel('Contact probability (100 kb bins)')
  ax.xaxis.set_ticks(x_range)
  ax.set_xticklabels(['$10^{%.1f}$' % x for x in x_range], fontsize=12)
  ax.set_xlim((np.log10(bin_size), np.log10(x_limit)))

  y_min, y_max = -9.5, -5.5
  y_range = np.arange(y_min, y_max, 0.5)
  ax.yaxis.set_ticks(y_range)
  ax.set_yticklabels(['$10^{%.1f}$' % x for x in y_range], fontsize=12)
  ax.set_ylim((y_min, y_max))

  ax.plot([5.5, 7.0], [-6.0, -7.50], color='#808080', alpha=0.5, linestyle='--')
  ax.plot([5.5, 7.0], [-6.5, -8.75], color='#808080', alpha=0.5, linestyle='--')
  ax.text(7.0, -7.50, '$\lambda=1.0$', color='#808080', verticalalignment='center', alpha=0.5, fontsize=14)
  ax.text(7.0, -8.75, '$\lambda=1.5$', color='#808080', verticalalignment='center', alpha=0.5, fontsize=14)

  ax.legend()

  plt.savefig(svg_path)
  plt.show()
  
  
  

def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument('-i', metavar='CONTACT_FILES', nargs='+',
                         help='Input NPZ or NCC format chromatin contact file(s). Wildcards accepted')

  arg_parse.add_argument('-o', metavar='PDF_FILE',
                         help='Output PDF format file. If not specified, a default based on the input file name(s).')

  arg_parse.add_argument('-g', default=False, action='store_true',
                         help='Display graphics on-screen using matplotlib, where possible and do not automatically save output.')

  arg_parse.add_argument('-i2', metavar='CONTACT_FILES', nargs='*',
                         help='Second group of input NPZ or NCC format chromatin contact file(s). Wildcards accepted')

  arg_parse.add_argument('-i3', metavar='CONTACT_FILES', nargs='*',
                         help='Third group of input NPZ or NCC format chromatin contact file(s). Wildcards accepted')

  arg_parse.add_argument('-i4', metavar='CONTACT_FILES', nargs='*',
                         help='Fourth group of input NPZ or NCC format chromatin contact file(s). Wildcards accepted')

  arg_parse.add_argument('-l', metavar='LABELS', nargs='*',
                         help='Text labels for groups of input files')

  arg_parse.add_argument('-s', default=100, metavar='KB_BIN_SIZE', type=int,
                         help='When using NCC format input, the sequence region size in kilobases for calculation of contact probabilities. Default is %d (kb)' % DEFAULT_BIN_SIZE)


 
  args = vars(arg_parse.parse_args(argv))

  in_paths1 = args['i']
  in_paths2 = args['i2'] or None
  in_paths3 = args['i3'] or None
  in_paths4 = args['i4'] or None
  pdf_path = args['o']
  bin_size = args['s']
  labels = args['l'] or None
  
  for paths in (in_paths1, in_paths2, in_paths3, in_paths4):
    if paths:
      for file_path in paths:
        invalid_msg = io.check_invalid_file(file_path)
        if invalid_msg:
          util.critical(invalid_msg)
 
  in_paths = [x for x in (in_paths1, in_paths2, in_paths3, in_paths4) if x]
    
  contact_probability(in_paths, pdf_path, bin_size, labels)
  
  # Add --regions
  # - Plot contact probability within and between regions
  #   - Inter/intra Enrichment vs seq separation
  #     + Within/between count ratio compared to permutation null 
  #     + Bootstrap errors
  #   - Plot distributions of partitioning inter/intra for each input
  #     + Each point/bin has an intra/inter ratio 
  #     + Histogram of ratios at different seq sep thresholds
  #     + Scatter of ratios for two samples
  # ? Do seq sep analysis too?
  

if __name__ == "__main__":
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  main()
