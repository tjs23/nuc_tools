import os, sys, csv
import numpy as np
from os.path import dirname
from glob import glob

sys.path.append(dirname(dirname(__file__)))

from nuc_tools import util
from formats import n3d, ncc

from matplotlib import pyplot as plt

def param_analysis(n3d_paths, ncc_path, tsv_path='structure_viol_analysis.tsv'):
  
  if not n3d_paths:
    print('No file paths')
  
  chromos, chromo_limits, contact_dict = ncc.load_file(ncc_path)
  
  sort_list = []
  
  n = len(n3d_paths)
  
  n_bins = 4
  n_models = None
  
  
  with open(tsv_path, 'w') as file_obj:
    write = file_obj.write
  
    for i, n3d_path in enumerate(n3d_paths):
      seq_pos_dict, coords_dict = n3d.load_n3d_coords(n3d_path)
      util.info(f'{i:,} of {n:,}', line_return=True)
      n_cont = 0
 
      if not n_models:
        n_models = coords_dict[list(coords_dict)[0]].shape[0]
        d_bins = np.zeros((n_models, n_bins))
 
      #fig, ax = plt.subplots()
      #hist = np.zeros(100, float)
 
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
 
        nm, a, d = coords_dict[chr_a].shape
 
        for m in range(n_models):
          coords_a = coords_dict[chr_a][m,idx_a]
          coords_b = coords_dict[chr_b][m,idx_b]

          deltas = coords_a - coords_b
          dists = np.sqrt((deltas * deltas).sum(axis=-1))
          dists = np.log10(dists+1)
 
          d_bins[m,0] += np.count_nonzero(dists <= 0.25)
          d_bins[m,1] += np.count_nonzero((dists > 0.25) & (dists <= 0.5))
          d_bins[m,2] += np.count_nonzero((dists > 0.50) & (dists <= 1.0))
          d_bins[m,3] += np.count_nonzero((dists > 1.00) & (dists <= 2.0))
 
          #h, edges = np.histogram(np.log10(dists), range=(0.0,2.0), bins=100)
          #hist += h
 
      #hist /= hist.sum()
      #ax.plot(edges[:-1], hist)
      #plt.show()
 
      for m in range(n_models):
        d_bins[m] *= 100.0/d_bins[m].sum() # Percentage for each dist class, per model
        
        values = [f'{x:5.2f}' for x in d_bins[m]]
        row = [n3d_path, str(m)] + values
        
        write('\t'.join(row) + '\n')

        
def plot_viol_analysis(tsv_path, lw=0.5):
  
  counts1 = []
  counts2 = []
  counts3 = []
  counts4 = []
  structs = []
  
  with open(tsv_path) as file_obj:
    
    for line in file_obj:
      struc, model, c1, c2, c3, c4 = line.rstrip().split()
      
      counts1.append(float(c1))
      counts2.append(float(c2))
      counts3.append(float(c3))
      counts4.append(float(c4))
      structs.append((struc, model))

  counts1 = np.array(counts1)
  counts2 = np.array(counts2)
  counts3 = np.array(counts3)
  counts4 = np.array(counts4)
  
  best = counts4.argmin()
  print('Best', structs[best])
  
  #counts1 -= counts1.mean()
  #counts2 -= counts2.mean()
  #counts3 -= counts3.mean()
  #counts4 -= counts4.mean()
  
  idx = counts4.argsort()
  
  counts1 = counts1[idx]
  counts2 = counts2[idx]
  counts3 = counts3[idx]
  counts4 = counts4[idx]
  
  fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
  
  ax1.set_title(tsv_path)
  
  ax1.plot(counts1, color='#808080', linewidth=lw, label='OK')
  ax2.plot(counts2, color='#0080FF', linewidth=lw, label='Low viol')
  ax3.plot(counts3, color='#B0B000', linewidth=lw, label='Med viol')
  ax4.plot(counts4, color='#FF0000', linewidth=lw, label='High viol')#
  
  ax1.legend(fontsize=9)
  ax2.legend(fontsize=9)
  ax3.legend(fontsize=9)
  ax4.legend(fontsize=9)
  

ncc_path = '/data/hi-c/SLX-22538_OH_HP1bATACsee2iL_P119E6_filter.ncc'

n3d_paths = glob('/data/hi-c/rotations_image_sched_B/P119E6_DNA_CEN_REPS*_imagX1_sched01_rots*_200.n3d')

tsv_path='structure_viol_analysis_rotation_search_200k_REPS_imagX1.tsv'
if not os.path.exists(tsv_path):
  param_analysis(n3d_paths, ncc_path, tsv_path)

plot_viol_analysis(tsv_path)

plt.show()
