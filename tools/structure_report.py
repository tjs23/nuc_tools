import os, sys
import numpy as np
from os.path import dirname

PROG_NAME = 'structure_report'
VERSION = '0.1.0'
DESCRIPTION = 'Generate a simple textual report for structure coordinates stored in multiple in N3D format files'

def structure_report(n3d_paths):

  from nuc_tools import util, io
  from formats import n3d

  util.info('Loading and aligining {:,} structures'.format(len(n3d_paths)))
  
  file_names = [os.path.basename(x) for x in n3d_paths]
  
  indent = max([len(x) for x in file_names])
  
  fn_format = '%{}.{}s'.format(indent, indent)
  
  head0 = [' ' * indent,'      ','    ','      ','   ----- Coordinate RMSDs ----']
  
  print('\t'.join(head0))

  head = [fn_format % 'File','p_size','n_chr','n_coord','    p50','     m0','    m50','   m100']
  
  print('\t'.join(head))
  
  data = []
  
  for i, n3d_path in enumerate(n3d_paths):
    seq_pos_dict, coords_dict = n3d.load_n3d_coords(n3d_path)
    coords_dict, rmsd_mat, model_mean_rmsds, particle_rmsds = util.align_chromo_coords(coords_dict, seq_pos_dict, dist_scale=False)
    
    model_rmsds = rmsd_mat.ravel()
    chromos = util.sort_chromosomes(coords_dict.keys())
    chromo = chromos[0]
    
    bin_size = int((seq_pos_dict[chromo][1] - seq_pos_dict[chromo][0])/1e3)
    
    n_chromos = len(chromos)
    n_particles = len(particle_rmsds)
    med_particle_rmsd = np.median(particle_rmsds)
    med_model_rmsd = np.median(model_rmsds)
    min_model_rmsd = np.min(model_rmsds)
    max_model_rmsd = np.max(model_rmsds)
    
    sort_key = (bin_size, med_model_rmsd, n_particles)
    file_name = fn_format % file_names[i]
    
    row = (file_name, bin_size, n_chromos, n_particles, med_particle_rmsd, min_model_rmsd, med_model_rmsd, max_model_rmsd)
    
    data.append((sort_key, row))
  
  data.sort()
  
  for sort_key, row in data:
    line = '{}\t{:6d}\t{:>5,}\t{:>7,}\t{:7.3f}\t{:7.3f}\t{:7.3f}\t{:7.3f}'.format(*row)
    
    print(line)
  
 
def main(argv=None):

  from argparse import ArgumentParser
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(metavar='N3D_FILES', nargs='+', dest='i',
                         help='One or more genome structure files in N3D format. Accepts wildcards.')

  args = vars(arg_parse.parse_args(argv))
  
  struc_paths = args['i']

  if not struc_paths:
    arg_parse.print_help()
    sys.exit(1)  
    
  structure_report(struc_paths)
  


if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()
