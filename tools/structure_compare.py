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
DEFAULT_RMSD_MAX = 3.5

def plot_structure_compare(seq_pos_dict1, coords_dict1, struc_label1,
                           seq_pos_dict2, coords_dict2, struc_label2,
                           bin_size, vmax, cmap, pdf):
                           
  
  from nuc_tools import util                        
  
  # Unify to common chromosome regions
  
  chromos = sorted(set(seq_pos_dict1) & set(seq_pos_dict2))
  coord_models1 = []
  coord_models2 = []
  #seq_pos_dict = {}
 # coords_dict = {}
  
  for chromo in chromos:
    a1 = seq_pos_dict1[chromo][0]
    b1 = seq_pos_dict1[chromo][-1]
    a2 = seq_pos_dict2[chromo][0]
    b2 = seq_pos_dict2[chromo][-1]
   
    a = max(a1, a2) 
    b = min(b1, b2) 
  
    i1 = (a-a1)//bin_size
    j1 = (b-a1)//bin_size
    i2 = (a-a2)//bin_size
    j2 = (b-a2)//bin_size
 
    coord_models1.append(coords_dict1[chromo][:,i1:j1])
    coord_models2.append(coords_dict2[chromo][:,i2:j2])
    #seq_pos_dict[chromo] = seq_pos_dict1[chromo][i1:j1]
 
  coord_models1 = np.concatenate(coord_models1, axis=1)
  coord_models2 = np.concatenate(coord_models2, axis=1)
 
  for i, coords in enumerate(coord_models1):
    coord_models1[i] = util.center_coords(coords)

  for i, coords in enumerate(coord_models2):
    coord_models2[i] = util.center_coords(coords)
   
  r1 = np.sqrt((coord_models1 * coord_models1).sum(axis=2)).mean()
  r2 = np.sqrt((coord_models2 * coord_models2).sum(axis=2)).mean()
  scale_ratio = r1/r2
  
  if not (0.9 < scale_ratio < 1.1):
    util.warn('Structures {} and {} have somewhat different sizes.'.format(struc_label1, struc_label2))
    coord_models2 *= scale_ratio

  n_models1 = len(coord_models1)
  n_models2 = len(coord_models2) 
  n_models = n_models1 + n_models2
    
  coord_models =  np.concatenate([coord_models1, coord_models2], axis=0)
  
  """
  i = 0
  for chromo in chromos:
    n = len(seq_pos_dict[chromo])
    coords_dict[chromo] = coord_models[:, i:i+n]
    i += n
  """
  
  rmsd_mat = np.zeros((n_models, n_models))
  
  for i in range(n_models-1):
    for j in range(i+1, n_models):
       coords_a, coords_b = util.align_coord_pair(coord_models[i], coord_models[j], 10.0)
       #coords_a, coords_b = util.align_coord_pair(coords_a, coords_b)
       
       model_rmsds, particle_rmsds = util.calc_rmsds(coords_a, [coords_b])
       rmsd = model_rmsds[0]
       rmsd_mat[i, j] = rmsd
       rmsd_mat[j, i] = rmsd

  
  mean_rmsd = rmsd_mat[:n_models1,n_models1:].mean()
  
  # Plot
  
  fig = plt.figure()
  fig.set_size_inches(max(5, n_models*0.25), max(5, n_models*0.25))
 
  plt.suptitle('Structure comparison\nA: {} B: {}'.format(struc_label1, struc_label2), fontsize=9)
  
  ax1 = fig.add_axes([0.05, 0.2, 0.22, 0.62])
  ax2 = fig.add_axes([0.27, 0.2, 0.62, 0.62])
  
  model_labels  = ['%2dA' % (i+1,) for i in range(n_models1)]
  model_labels += ['%2dB' % (i+1,) for i in range(n_models2)]

  linkage = hierarchy.linkage(distance.squareform(rmsd_mat), method='ward', optimal_ordering=True)
  order= hierarchy.leaves_list(linkage)[::-1]
  model_labels = [model_labels[i] for i in order]
  
  ddict = hierarchy.dendrogram(linkage, orientation='left', labels=model_labels,
                               above_threshold_color='#808080',
                               link_color_func=lambda k: '#808080', ax=ax1)
  
  ax1.set_xticklabels([])
  ax1.set_xticks([])
  ax1.set_axis_off()
                               
  img2 = ax2.matshow(rmsd_mat[order][:,order], cmap=cmap,
                     aspect='auto', vmin=0.0, vmax=vmax)
 
  ax2.text(0.27, 0.84, 'Model-Model RMSD $\mu_{AB}$=%.2f' % mean_rmsd,
           fontsize=7, transform=fig.transFigure, ha='left')  
  
  ax2.tick_params(which='both', direction='out', left=False, right=True,
                 labelright=True, labelleft=False, pad=8)
  ax2.set_yticklabels(model_labels, fontsize=7, ha='center')
  ax2.set_xticklabels([])
  ax2.set_xticks([])
  ax2.set_yticks(np.arange(0, n_models))
 
  axcb = fig.add_axes([0.27, 0.15, 0.62, 0.02])
  
  cbar = plt.colorbar(img2, cax=axcb, orientation='horizontal')
  cbar.ax.tick_params(labelsize=7)
  cbar.set_label('RMSD', fontsize=7) 
    
  if pdf:
    pdf.savefig(dpi=PDF_DPI)
  else:
    plt.show() 
  
  plt.close()
  
  #return coords_dict, seq_pos_dict
  
 
def plot_structure_report(label, seq_pos_dict, rmsd_mat, model_mean_rmsds, particle_rmsds, vmax, cmap, pdf):
  
  from nuc_tools import util
  from matplotlib.ticker import AutoMinorLocator
  
  n_models = len(rmsd_mat)
  n_chromos = len(seq_pos_dict)
  
  mean_rmsd = np.mean(rmsd_mat[np.triu_indices(n_models, k=1)]) # Above diagonal
  
  fig = plt.figure()
  fig.set_size_inches(max(5, n_models*0.5), max(5, n_models*0.5))
  
  plt.suptitle('Coordinate superposition {}'.format(label), fontsize=9)
  
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
    r, g, b, a = cmap(min(vmax, rmsd/vmax))
    grey = (0.5 + ((0.3 * r) + (0.59 * g) + (0.11 * b))) % 1.0
    color = (grey, grey, grey, 1.0)
    ax3.text(0.0, float(n_models-i-1), '%.2f' % rmsd, fontsize=5, va='center', ha='center', color=color)
  
  cbar = plt.colorbar(img2, cax=axcb)
  cbar.ax.tick_params(labelsize=7)
  cbar.set_label('RMSD', fontsize=7) 
  
  chromos = util.sort_chromosomes(seq_pos_dict.keys())
  bin_size = seq_pos_dict[chromos[0]][1] - seq_pos_dict[chromos[0]][0]
  chromos.reverse()
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
 

def check_file_labels(file_paths, labels):
  
  from nuc_tools import io
  
  if labels:
    n_files = len(file_paths)
    for i, label in enumerate(labels):
      labels[i] = label.replace('_',' ')
      
    while len(labels) < n_files:
      i = len(labels)
      labels.append(io.get_file_root(file_paths[i]))
      
  else:
    labels = [io.get_file_root(x) for x in file_paths]
  
  return labels
  
 
def structure_compare(struc_paths1, struc_labels1, struc_paths2, struc_labels2,
                      bin_size, out_pdf_path, screen_gfx, rmsd_max, cmap):
  
  from nuc_tools import util, io
  from formats import n3d
  
  struc_paths = struc_paths1 + struc_paths2
  n_strucs = len(struc_paths)
    
  if out_pdf_path:
    out_path = io.check_file_ext(out_pdf_path, '.pdf')
  
  else:
    out_path = io.get_out_job_file_path(struc_paths[0], DEFAULT_PDF_OUT, [n_strucs])  
   
  struc_labels1 = check_file_labels(struc_paths1, struc_labels1)
  struc_labels2 = check_file_labels(struc_paths2, struc_labels2)
  struc_labels = struc_labels1 + struc_labels2
   
  if screen_gfx:
    pdf = None
  else:
    pdf = PdfPages(out_path)  
  
  for i, struc_n3d_path in enumerate(struc_paths):
    util.info('Loading and aligining {}'.format(struc_n3d_path))
    seq_pos_dict, coords_dict = n3d.load_n3d_coords(struc_n3d_path)
    coords_dict, rmsd_mat, model_mean_rmsds, particle_rmsds = util.align_chromo_coords(coords_dict, seq_pos_dict, dist_scale=False)
    plot_structure_report(struc_labels[i], seq_pos_dict, rmsd_mat, model_mean_rmsds, particle_rmsds, rmsd_max, cmap, pdf)
    util.info('  .. found {} models with {:,} coords for {} chromosomes'.format(len(model_mean_rmsds), len(particle_rmsds), len(coords_dict)))
  
  if struc_paths2:
    for i, struc_n3d_path1 in enumerate(struc_paths1):
      seq_pos_dict1, coords_dict1 = n3d.load_n3d_coords(struc_n3d_path1)
 
      for j, struc_n3d_path2 in enumerate(struc_paths2):
         util.info('Comparing {} with {}'.format(struc_n3d_path1, struc_n3d_path2))
         seq_pos_dict2, coords_dict2 = n3d.load_n3d_coords(struc_n3d_path2)
 
         plot_structure_compare(seq_pos_dict1, coords_dict1, struc_labels1[i],
                                seq_pos_dict2, coords_dict2, struc_labels2[j],
                                bin_size, rmsd_max, cmap, pdf)
 
         #coords_dict, rmsd_mat, model_mean_rmsds, particle_rmsds = util.align_chromo_coords(coords_dict, seq_pos_dict, dist_scale=False)
         #plot_structure_report('Both', seq_pos_dict, rmsd_mat, model_mean_rmsds, particle_rmsds, rmsd_max, cmap, pdf)
         
  if pdf:
    pdf.close()
    util.info('Written {}'.format(out_path))
  else:
    util.info('Done')
  
  
def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  from formats import n3d
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(metavar='N3D_FILES', nargs='+', dest='i',
                         help='One or more primary genome structure files in N3D format. Accepts wildcards.')

  arg_parse.add_argument('-i2', metavar='N3D_FILES', nargs='*', dest='i2',
                         help='One or more secondary genome structure files in N3D format. All primary structures will ' \
                              'be compared directly to all secondary structures, Accepts wildcards.')
 
  arg_parse.add_argument('-l', '--sruc-labels', metavar='STRUCTURE_NAMES', nargs='+', dest='l',
                         help='Optional textual labels/names for the input structures.')

  arg_parse.add_argument('-l2', '--sruc-labels-secondary', metavar='STRUCTURE_NAMES', nargs='+', dest='l2',
                         help='Optional textual labels/names for the secondary input structures.')

  arg_parse.add_argument('-g', '--screen-gfx', default=False, action='store_true', dest='g',
                         help='Display graphics on-screen using matplotlib, where possible and ' \
                              'do not automatically save graphical output to file.')
                              
  arg_parse.add_argument('-o', '--out-pdf', metavar='OUT_PDF_FILE', default=None, dest='o',
                         help='Optional output PDF file name. If not specified, a default will be used.')

  arg_parse.add_argument('-mx', '--rmsd-limit', metavar='RMSD', default=DEFAULT_RMSD_MAX, type=float,  dest='mx',
                         help='Maximum coordinate RMSD value used in plot scales. Values greater than this will be clipped. Default: %.3f.' % DEFAULT_RMSD_MAX)

  arg_parse.add_argument('-colors', metavar='COLOR_SCALE', default='#FFFFFF,#0080FF,#D00000',
                         help='Optional scale colours as a comma-separated list, e.g. "white,blue,red".' \
                              'or colormap (scheme) name, as used by matplotlib. ' \
                              'Note: #RGB style hex colours must be quoted e.g. "#FF0000,#0000FF" ' \
                              'See: %s This option overrides -b.' % util.COLORMAP_URL)

  args = vars(arg_parse.parse_args(argv))
  
  struc_paths = args['i']
  struc_paths2 = args['i2'] or []
  struc_labels = args['l'] or []
  struc_labels2 = args['l2'] or []
  out_pdf_path = args['o']
  screen_gfx = args['g']
  rmsd_max = args['mx']
  cmap = args['colors']
   
  if not struc_paths:
    arg_parse.print_help()
    sys.exit(1)  
  
  bin_size = None
  for in_path in struc_paths + struc_paths2:
    io.check_invalid_file(in_path)  
    bin_size2 = n3d.get_bin_size(in_path)
    
    if bin_size is None:
      bin_size = bin_size2
    
    elif bin_size2 != bin_size:
      msg = 'Input N3D format coordinate files must have the same chromosome bin/particle size. ' \
            'Found {:,} bp and {:,} bp sizes.'
      util.critical(msg.format(bin_size, bin_size2))
 
  if cmap:
    cmap = util.string_to_colormap(cmap)
  
  nl = len(struc_labels)
  ns = len(struc_paths)
  if nl and  nl > ns:
    util.warn('Number of structure labels (%d) exceeds the number of structure files (%d)' % (nl, nd))
    struc_labels = struc_labels[:ns]

  nl = len(struc_labels2)
  ns = len(struc_paths2)
  if nl and  nl > ns:
    util.warn('Number of secondary structure labels (%d) exceeds the number of secondary structure files (%d)' % (nl, nd))
    struc_labels2 = struc_labels2[:ns]
  
  structure_compare(struc_paths, struc_labels, struc_paths2, struc_labels2,
                    bin_size, out_pdf_path, screen_gfx, rmsd_max, cmap)
  
if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()

"""
/data/hi-c/calc_25k/n3d_25/SLX-15484_INLINE_HTFK2BBXX_s_8_r_1_2_P64E5_25k.n3d
./nuc_tools structure_compare /home/tjs23/gh/nuc_tools_bak/n3d/Cell*_100kb_x10.n3d -o /home/tjs23/Desktop/test_struc_comp.pdf

"""

