import os, sys
import numpy as np
from os.path import dirname

PROG_NAME = 'data_track_compare'
VERSION = '1.0.0'
DESCRIPTION = 'Plot and measure similarities between data tracks in BED format'

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Colormap

DEFAULT_PDF_OUT = 'dtc_out_job{}_D{}.pdf'
DEFAULT_BIN_KB  = 10.0

def data_track_compare():
  
  
  
  
def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(metavar='BED_FILES', nargs='+', dest='d',
                         help='Primary input data track files to be compared, in BED format. All data tracks will be compared to all others inless the -d is used.')

  arg_parse.add_argument('-l', '--data-labels', metavar='DATA_NAMES', nargs='+', dest='l',
                         help='Optional textual labels/names for the primary input data tracks.')

  arg_parse.add_argument('-d2', '--data-files2', metavar='BED_FILES', nargs='+', dest='d2',
                         help='Secondary input data track files to be compared, in BED format. All primary data wil lbe compare dto all secondary data.')

  arg_parse.add_argument('-l2', '--data-labels', metavar='DATA_NAMES', nargs='+', dest='l2',
                         help='Optional textual labels/names for the secondary input data tracks.')

  arg_parse.add_argument('-o', '--out-pdf', metavar='OUT_PDF_FILE', default=None, dest='o',
                         help='Optional output PDF file name. If not specified, a default of the form %s will be used.' % DEFAULT_PDF_OUT.format('{#}','{#}'))

  arg_parse.add_argument('-s', '--bin-size', default=None, metavar='BIN_SIZE', type=float, dest="s1",
                         help='Binned sequence region size, in kilobases: data tracks are compared across equal sized chromosome regions.' \
                              'Default is {:.1f} kb .'.format(DEFAULT_BIN_KB))
  
  arg_parse.add_argument('-bp', '--boxplot', default=False, action='store_true', dest='bp',
                         help='Use boxplots (with outliers) for displays of certain data distributions rather than violin plots.')

  arg_parse.add_argument('-g', '--screen-gfx', default=False, action='store_true', dest='g',
                         help='Display graphics on-screen using matplotlib, where possible and ' \
                              'do not automatically save graphical output to file.')

  args = vars(arg_parse.parse_args(argv))
                                
  ref_data_paths = args['d']
  comp_data_paths = args['d2']
  ref_labels = args['l']
  comp_labels = args['l2']
  out_pdf_path = args['o']
  bin_size = args['s']
  screen_gfx = args['g']
  use_boxplots = args['bp']
  
  for in_path in ref_data_paths + comp_data_paths:
    io.check_invalid_file(in_path)
    
   
  data_track_compare(ref_data_paths, comp_data_paths, ref_labels, comp_labels, out_pdf_path, bin_size, screen_gfx, use_boxplots)
  
if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()

"""

Correlations between data values of region binned data 
 - Quote both Pearson CC (r) and Spearman rank (rho) and num vals

Non-quantile normalisation
 - basic scatter density plot

Quantile normalisation (for secondary axis)
 - basic scatter density plot
 - violin/boxplots

Combined plot (one for each primary)
 - Lines of quantile mean +/- std, +/- stderr for each secondary

Density matrix of all track similarities
 - correlations
 - sequence proximity

Plot for optimal bin size
 - for each pair correlation vs bin_size
   + combine smaller bins for speed
   + log scale for dynamic range

Plot of seq overlap
 - for each primary the distribution of closest secondary
   + both strands
 - stack colour matrix of all secondary signal centred on primary site
 
./nuc_tools ddata_track_compare /data/bed/H3K4me3_hap_EDL.bed -d2 /data/bed/H3K9me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed
"""

