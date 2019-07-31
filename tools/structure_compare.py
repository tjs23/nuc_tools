import os, sys, math
import numpy as np
from collections import defaultdict
from glob import glob


from os.path import dirname
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.cluster import hierarchy
from scipy.spatial import distance

PROG_NAME = 'structure_compare'
VERSION = '1.0.0'
DESCRIPTION = 'Compare genome/chromosome structure coordinates in N3D format'
              
DEFAULT_PDF_OUT = 'sc_out_job{}_S{}.pdf'
PDF_DPI = 200


def plot_rmsd_matrix(struc_data, struc_labels, cmap, pdf):
  
  n_strucs = len(struc_labels)
  
  struc_rmsds = [x[2] for x in struc_data]
   
  fig, axarr = plt.subplots(n_strucs, 2, squeeze=False)    
  fig.set_size_inches(4, 2*n_strucs)
  
  plt.suptitle('Model-model coordinate RMSDs')
  plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.15, hspace=0.1)
  
   
  for i, label in enumerate(struc_labels):
    rmsd_mat = struc_rmsds[i]
    n_models = len(rmsd_mat)
    model_labels = ['%d' % (x+1) for x in range(n_models)]
    
    ax1 = axarr[i, 0]
    ax2 = axarr[i, 1]
    
    linkage = hierarchy.linkage(distance.squareform(rmsd_mat), method='ward', optimal_ordering=True)
    order= hierarchy.leaves_list(linkage)[::-1]
    model_labels = [model_labels[i] for i in order]
    
    ddict = hierarchy.dendrogram(linkage, orientation='left', labels=model_labels,
                                 above_threshold_color='#000000',
                                 link_color_func=lambda k: '#000000', ax=ax1)
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_axis_off()
    
    cax = ax2.matshow(rmsd_mat[order][:,order], cmap=cmap, aspect='auto', vmin=0.0)
    
    ax2.set_title(label)
    ax2.set_xticklabels([])
    ax2.set_xticks([])
    ax2.set_yticklabels(model_labels, fontsize=7)
    ax2.set_yticks(np.arange(0, n_models))
   
  
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show() 
  
  plt.close()

def structure_compare(struc_paths, struc_labels, out_pdf_path, screen_gfx, cmap):
  
  from nuc_tools import util, io
  from formats import n3d
  
  n_strucs = len(struc_paths)
    
  if out_pdf_path:
    out_path = io.check_file_ext(out_pdf_path, '.pdf')
  
  else:    
    dir_path = dirname(struc_paths[0])
    
    job = 1
    while glob(os.path.join(dir_path, DEFAULT_PDF_OUT.format(job, '*', '*'))):
      job += 1
    
    file_name = DEFAULT_PDF_OUT.format(job, n_structs)
    out_pdf_path = os.path.join(dir_path, file_name)    
  
  if struc_labels:
    for i, label in enumerate(struc_labels):
      struc_labels[i] = label.replace('_',' ')
      
    while len(struc_labels) < n_strucs:
      i = len(struc_labels)
      struc_labels.append(io.get_file_root(struc_paths[i]))
      
  else:
    struc_labels = [io.get_file_root(x) for x in struc_paths]
  
  struc_data = []
  
  for struc_n3d_path in struc_paths:
    util.info('Loading and aligining {}'.format(struc_n3d_path))
    seq_pos_dict, coords_dict = n3d.load_n3d_coords(struc_n3d_path)
    coords_dict, model_rmsds, model_mean_rmsds, particle_rmsds = util.align_chromo_coords(coords_dict, seq_pos_dict)
    
    util.info('  .. found {} models with {:,} coords for {} chromosomes'.format(len(model_mean_rmsds), len(particle_rmsds), len(coords_dict)))
    struc_data.append((seq_pos_dict, coords_dict, model_rmsds, model_mean_rmsds, particle_rmsds))
 
  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(out_path)  
  
  # Separate plot for each structure
  # - rmsd vs seq per chromo
  # - model rmsds, overall RMSD
  
  # Combined plots
  # - median model RMSD
  
  plot_rmsd_matrix(struc_data, struc_labels, cmap, pdf) 
       
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
  
  arg_parse.add_argument(metavar='N3D_FILES', nargs='+', dest='i',
                         help='One or more genome structure files in N3D format. Accepts wildcards.')
 
  arg_parse.add_argument('-l', '--sruc-labels', metavar='STRUCTURE_NAMES', nargs='+', dest='l',
                         help='Optional textual labels/names for the input structures.')

  arg_parse.add_argument('-g', '--screen-gfx', default=False, action='store_true', dest='g',
                         help='Display graphics on-screen using matplotlib, where possible and ' \
                              'do not automatically save graphical output to file.')
                              
  arg_parse.add_argument('-o', '--out-pdf', metavar='OUT_PDF_FILE', default=None, dest='o',
                         help='Optional output PDF file name. If not specified, a default will be used.')

  arg_parse.add_argument('-colors', metavar='COLOR_SCALE', default='w,b,y',
                         help='Optional scale colours as a comma-separated list, e.g. "white,blue,red".' \
                              'or colormap (scheme) name, as used by matplotlib. ' \
                              'Note: #RGB style hex colours must be quoted e.g. "#FF0000,#0000FF" ' \
                              'See: %s This option overrides -b.' % util.COLORMAP_URL)

  args = vars(arg_parse.parse_args(argv))
  
  struc_paths = args['i']
  struc_labels = args['l'] or []
  out_pdf_path = args['o']
  screen_gfx = args['g']
  cmap = args['colors']
   
  if not struc_paths:
    arg_parse.print_help()
    sys.exit(1)  
  
  for in_path in struc_paths:
    io.check_invalid_file(in_path)  
 
  if cmap:
    cmap = util.string_to_colormap(cmap)
  
  nl = len(struc_labels)
  ns = len(struc_paths)
  if nl and  nl > ns:
    util.warn('Number of strucre labels (%d) exceeds the number of structure files (%d)' % (nl, nd))
    data_labels = data_labels[:ns]
  
  structure_compare(struc_paths, struc_labels, out_pdf_path, screen_gfx, cmap)
  
if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()

"""
/data/hi-c/calc_25k/n3d_25/SLX-15484_INLINE_HTFK2BBXX_s_8_r_1_2_P64E5_25k.n3d
./nuc_tools structure_compare /home/tjs23/gh/nuc_tools_bak/n3d/Cell*_100kb_x10.n3d -o test_sc.pdf

"""

