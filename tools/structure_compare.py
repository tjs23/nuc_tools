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

def plot_structure_report(label, seq_pos_dict, rmsd_mat, model_mean_rmsds, particle_rmsds, cmap, pdf):
  
  from nuc_tools import util
  from matplotlib.ticker import AutoMinorLocator
  
  vmax = 3.5

  n_models = len(rmsd_mat)
  n_chromos = len(seq_pos_dict)
  
  mean_rmsd = np.mean(rmsd_mat[np.triu_indices(n_models, k=1)]) # Above diagonal
  
  fig = plt.figure()
  fig.set_size_inches(max(5, n_models*0.5), max(5, n_models*0.5))
  
  plt.suptitle('Structure report {}'.format(label), fontsize=9)
  
  ax1 = fig.add_axes([0.07, 0.5, 0.23, 0.4])
  ax2 = fig.add_axes([0.30, 0.5, 0.4, 0.4])
  ax3 = fig.add_axes([0.77, 0.5, 0.07, 0.4])
  ax4 = fig.add_axes([0.1, 0.07, 0.85, 0.38])
  
  axcb = fig.add_axes([0.86, 0.50, 0.02, 0.3])
 
  model_labels = ['%2d' % (i+1,) for i in range(n_models)]
  
  linkage = hierarchy.linkage(distance.squareform(rmsd_mat), method='ward', optimal_ordering=True)
  order= hierarchy.leaves_list(linkage)[::-1]
  model_labels = [model_labels[i] for i in order]
  
  ddict = hierarchy.dendrogram(linkage, orientation='left', labels=model_labels,
                               above_threshold_color='#808080',
                               link_color_func=lambda k: '#808080', ax=ax1)
  
  ax1.text(0.05, 0.76, 'Coord model',
           fontsize=7, transform=fig.transFigure, ha='center', rotation=90)  
  
  ax1.set_xticklabels([])
  ax1.set_xticks([])
  ax1.set_axis_off()

  img2 = ax2.matshow(rmsd_mat[order][:,order], cmap=cmap,
                     aspect='auto', vmin=0.0, vmax=vmax)
 
  ax2.text(0.3, 0.91, 'Model-Model RMSD $\mu$=%.2f' % mean_rmsd,
           fontsize=7, transform=fig.transFigure, ha='left')  
  
  ax2.tick_params(which='both', direction='out', left=False, right=True,
                 labelright=True, labelleft=False, pad=8)
  ax2.set_yticklabels(model_labels, fontsize=7, ha='center')
  ax2.set_xticklabels([])
  ax2.set_xticks([])
  ax2.set_yticks(np.arange(0, n_models))
  
  img3 = ax3.matshow(model_mean_rmsds[::-1].reshape(n_models, 1), cmap=cmap,
                     aspect='auto', vmin=0.0, vmax=vmax)
  
  ax3.text(0.81, 0.91, 'Coord $\mu$', fontsize=7, transform=fig.transFigure, ha='center')  
  
  ax3.set_yticklabels([])
  ax3.set_xticklabels([])
  ax3.set_xticks([])
  ax3.set_yticks(np.arange(0, n_models))
  
  for i, rmsd in enumerate(model_mean_rmsds):
    ax3.text(0.0, float(n_models-i-1), '%.2f' % rmsd, fontsize=5, va='center', ha='center', color='#FFFFFF')
  
  cbar = plt.colorbar(img2, cax=axcb)
  cbar.ax.tick_params(labelsize=7)
  cbar.set_label('RMSD', fontsize=7) 
  
  chromos = util.sort_chromosomes(seq_pos_dict.keys())
  bin_size = seq_pos_dict[chromos[0]][1] - seq_pos_dict[chromos[0]][0]
  chromo_rmsds = {}
  
  n_max = 1
  i = 0
  for chromo in sorted(seq_pos_dict):
    n = len(seq_pos_dict[chromo])
    chromo_rmsds[chromo] = particle_rmsds[i:i+n]
    n_max = max(n_max, 1+int(seq_pos_dict[chromo].max()//bin_size))
    i += n
  
  x_max = (n_max * bin_size) / 1e6
  
  chromo_labels = []
  chromo_mat = np.zeros((n_chromos, n_max))
  
  for i, chromo in enumerate(chromos):
    label = chromo[3:] if chromo.lower().startswith('chr') else chromo
    chromo_labels.append(label)
    rmsds = chromo_rmsds[chromo]
    start = int(seq_pos_dict[chromo][0]//bin_size)
    chromo_mat[i,start:start+len(rmsds)] = rmsds
  
  ax4.text(0.5, 0.46, 'Particle RMSD', fontsize=7, transform=fig.transFigure, ha='center')  
  ax4.matshow(chromo_mat, cmap=cmap,
             aspect='auto', vmin=0.0, vmax=vmax)
 
  ax4.spines["right"].set_visible(False)
  ax4.spines["top"].set_visible(False)
  ax4.xaxis.set_minor_locator(AutoMinorLocator())
  
  ax4.set_xlabel('Position (Mb)', fontsize=7)
  ax4.tick_params(axis='both', which='both', labelsize=5, pad=2, top=False, bottom=True,
                 labelbottom=True, labeltop=False,)
  
  ax4.set_yticklabels(chromo_labels)
  ax4.set_yticks(np.arange(n_chromos))
  
  ax4.set_ylabel('Chromosome', fontsize=7)
    
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
 
  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(out_path)  
  
  struc_data = []
  
  for i, struc_n3d_path in enumerate(struc_paths):
    util.info('Loading and aligining {}'.format(struc_n3d_path))
    seq_pos_dict, coords_dict = n3d.load_n3d_coords(struc_n3d_path)
    coords_dict, rmsd_mat, model_mean_rmsds, particle_rmsds = util.align_chromo_coords(coords_dict, seq_pos_dict, dist_scale=False)
    
    plot_structure_report(struc_labels[i], seq_pos_dict, rmsd_mat, model_mean_rmsds, particle_rmsds, cmap, pdf)
     
    util.info('  .. found {} models with {:,} coords for {} chromosomes'.format(len(model_mean_rmsds), len(particle_rmsds), len(coords_dict)))
    struc_data.append((seq_pos_dict, coords_dict, rmsd_mat, model_mean_rmsds, particle_rmsds))
  
  # Separate plot for each structure
  # - rmsd vs seq per chromo
  # - model rmsds, overall RMSD
  
  # Combined plots
  # - median model RMSD
  
  #plot_rmsd_matrix(struc_data, struc_labels, cmap, pdf) 
  
       
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

  arg_parse.add_argument('-colors', metavar='COLOR_SCALE', default='#FFFFFF,#0080FF,#D00000',
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

