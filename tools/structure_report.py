import os, sys, csv
import numpy as np
from os.path import dirname

PROG_NAME = 'structure_report'
VERSION = '0.1.0'
DESCRIPTION = 'Generate a simple textual report for structure coordinates stored in multiple in N3D format files'

def structure_report(n3d_paths, ncc_paths, out_path=None):

  from nuc_tools import util, io
  from formats import n3d, ncc

  util.info('Loading and aligining {:,} structures'.format(len(n3d_paths)))
    
  file_names = [os.path.basename(x) for x in n3d_paths]
  
  #indent = max([len(x) for x in file_names])
  #fn_format = '%{}.{}s'.format(indent, indent)
  
  if out_path:
    file_obj = open(out_path, 'w')
    write = file_obj.write
    
  else:
    write = sys.stdout.write
    file_obj = None
  
  #head0 = [' ' * indent,'      ','    ','      ','    -------- Coordinate RMSDs -------']
  head = ['N3D_file','p_size_kb','n_chromo','n_models','n_coords','rmsd_mt50','rmsd_p50','rmsd_m0','rmsd_m50','rmsd_m100']
  
  if ncc_paths:
    #head0 += ['--- Distance violations ---']
    head  += ['NCC_file', 'n_contacts', 'viol_3', 'viol_4', 'viol_5']
  
  #write('\t'.join(head0))
  write('\t'.join(head) + '\n')
  
  data = []
  
  for i, n3d_path in enumerate(n3d_paths):
    seq_pos_dict, coords_dict = n3d.load_n3d_coords(n3d_path)
    
    # get violations
    coords_dict, rmsd_mat, model_mean_rmsds, particle_rmsds = util.align_chromo_coords(coords_dict, seq_pos_dict, dist_scale=False)
    
    m = len(rmsd_mat)
    best_idx = set(model_mean_rmsds.argsort()[:max(1, int(m/2))])
    
    model_rmsds = []
    best_model_rmsds = []
    for j in range(m-1):
      for k in range(j+1,m):
        model_rmsds.append(rmsd_mat[j,k])
        
        if j in best_idx and k in best_idx:
          best_model_rmsds.append(rmsd_mat[j,k])
        
    chromos = util.sort_chromosomes(coords_dict.keys())
    for chromo in chromos:
      if len(seq_pos_dict[chromo]) > 1:
        break
    
    bin_size = int((seq_pos_dict[chromo][1] - seq_pos_dict[chromo][0])/1e3)
    
    n_models = len(rmsd_mat)
    n_chromos = len(chromos)
    n_particles = len(particle_rmsds)
    med_particle_rmsd = np.median(particle_rmsds)
    med_model_rmsd = np.median(model_rmsds)
    min_model_rmsd = np.min(model_rmsds)
    max_model_rmsd = np.max(model_rmsds)
    best_med_rmsd = np.mean(best_model_rmsds)
    
    sort_key = (bin_size, best_med_rmsd, med_model_rmsd, n_particles)
    file_name = file_names[i]
    
    # - if matching NCC file; position pairs
    # - search sorted into positional pairs; bead pairs
    
    if ncc_paths:
      ncc_file = os.path.basename(ncc_paths[i])
      chromos, chromo_limits, contact_dict = ncc.load_file(ncc_paths[i])
      
      viol3 = 0
      viol4 = 0
      viol5 = 0
      n_cont = 0
 
      for chr_a, chr_b in contact_dict:
        if chr_a not in seq_pos_dict:
          continue
        if chr_b not in seq_pos_dict:
          continue
      
        contacts = contact_dict[(chr_a, chr_b)]
 
        seq_pos_a = contacts[:,0]
        seq_pos_b = contacts[:,1]
 
        idx_a = np.searchsorted(seq_pos_dict[chr_a], seq_pos_a)-1 
        idx_b = np.searchsorted(seq_pos_dict[chr_b], seq_pos_b)-1
 
        coords_a = coords_dict[chr_a][:,idx_a]
        coords_b = coords_dict[chr_b][:,idx_b]
 
        deltas = coords_a - coords_b
        dists = np.sqrt((deltas * deltas).sum(axis=2))
        model_median_dists = np.median(dists, axis=0)
 
        viol3 += np.count_nonzero(model_median_dists > 3.0)
        viol4 += np.count_nonzero(model_median_dists > 4.0)
        viol5 += np.count_nonzero(model_median_dists > 5.0)
        n_cont += contacts.shape[0]
      
      if n_cont:
        viol3 *= 100.0/float(n_cont)
        viol4 *= 100.0/float(n_cont)
        viol5 *= 100.0/float(n_cont)
   
      row = (file_name, bin_size, n_chromos, n_models,n_particles,
             best_med_rmsd, med_particle_rmsd, min_model_rmsd, med_model_rmsd,
             max_model_rmsd, ncc_file, n_cont, viol3, viol4, viol5)
    
    else: 
      row = (file_name, bin_size, n_chromos, n_models, n_particles, best_med_rmsd,
             med_particle_rmsd, min_model_rmsd, med_model_rmsd, max_model_rmsd)
    
    data.append((sort_key, row))
  
  #data.sort()
    
  line_fmt = '{}\t{}\t{}\t{}\t{}\t{:7.3f}\t{:7.3f}\t{:7.3f}\t{:7.3f}\t{:7.3f}\n'
  key_line = '\n-- Column Key --\nrmsd_mt50: Median model RMSD for top 50% of models\nrmsd_p50: Median particle RMSD\nrmsd_m0: Minimum model-model RMSD\n' \
             'rmsd_m50: Median model-model RMSD\nrmsd_m100: Maximum model-model RMSD\n'
  
  if ncc_paths:
    line_fmt = line_fmt[:-1] + '\t{}\t{}\t{:7.3f}\t{:7.3f}\t{:7.3f}\n'
    key_line = key_line + 'viol_3: Percent contact distances > 3.0 radii\nviol_4: Percent contact distances > 4.0 radii\nviol_5: Percent contact distances > 5.0 radii\n'
    
  for sort_key, row in data:
    write(line_fmt.format(*row))
  
  write(key_line)
  
  if file_obj:
    util.info('Written report to {}'.format(out_path))
    file_obj.close()
  
  """
  from matplotlib import pyplot as plt
  
  alpha = 5
  
  y_vals1 = [x[0][1] for x in data] # best_med_rmsd
  y_vals2 = [x[0][2] for x in data] # med_model_rmsd
  y_max = max(max(y_vals1), max(y_vals2))
  
  fig, (ax1, ax2) = plt.subplots(2, 1)
  
  y_max = int(y_max+1.0)
  bins = np.linspace(0, y_max, 4*y_max)
  
  ax1.plot(y_vals1, alpha=alpha, label='Median RMSD top models')
  ax1.plot(y_vals2, alpha=alpha, label='Median RMSD')
  ax1.set_xlabel('Structure')
  ax1.set_ylabel('Inter-model RMSD')
  ax1.legend()
  
  hist1, edges1 = np.histogram(y_vals1, bins=bins)
  hist2, edges2 = np.histogram(y_vals2, bins=bins)
  
  ax2.plot(edges1[:-1], hist1, alpha=alpha, label='Median RMSD top models')
  ax2.plot(edges2[:-1], hist2, alpha=alpha, label='Median RMSD')
  ax2.set_xlabel('Inter-model RMSD bin')
  ax2.set_ylabel('Count')
  ax2.legend()
  
  plt.show()
  """
  
def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(metavar='N3D_FILES', nargs='+', dest='i',
                         help='One or more genome structure files in N3D format. Accepts wildcards.')
  
  arg_parse.add_argument('-c', metavar='NCC_FILES', nargs='+', dest='c',
                         help='Optionally, one or more genome contact files in NCC format. Accepts wildcards. File order should match the corresponding N3D coordinate file.')

  arg_parse.add_argument('-o', metavar='TSV_OUT',  dest='o', default=None,
                         help='Output TSV format file path for table. Optional; if not set results are printed to screen.')

  args = vars(arg_parse.parse_args(argv))
  
  struc_paths = args['i']
  contact_paths = args['c']
  out_path = args['o']
  
  if not struc_paths:
    arg_parse.print_help()
    sys.exit(1)  
    
  
  if contact_paths and (len(contact_paths) != len(struc_paths)):
    util.critical('Number of input structure files ({}) does not match number of contact files ({})'.format(len(struc_paths), len(contact_paths)))
  
  structure_report(struc_paths, contact_paths, out_path)
  

if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()
