import sys, os, time
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from collections import defaultdict
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nuc_tools import util, io, parallel
from formats import bed, n3d

# Plot distribution of trans distances, aggregated

from numba import njit, float64, prange, int64

@njit(float64[:](float64[:,:,:],float64[:,:,:],int64[:],int64[:]), fastmath=True, parallel=True)
def dist_density(coords1, coords2, idx1, idx2):
  
  p, n, dim = coords1.shape
  q, m, dim = coords2.shape
  pf = float(p)
  
  densities = np.empty(n)
  
  ncpu = 12
  
  for cpu in prange(ncpu):
    for i in range(cpu, n, ncpu):
      mean_inv_d2 = 0.0
      p1 = idx1[i]
      
      for j in range(m):
        p2 = idx2[j]
        
        if -3 < (p1-p2) < 3: # Close on backbone
          continue
         
        d2 = 0.0
 
        for k in range(p): # model
 
          dx = coords1[k,i,0] - coords2[k,j,0]
          dy = coords1[k,i,1] - coords2[k,j,1]
          dz = coords1[k,i,2] - coords2[k,j,2]

          d2 += max(1.0, dx * dx + dy * dy + dz * dz)
        
        mean_inv_d2 += pf/d2
       
      densities[i] = mean_inv_d2 / m
   
  return densities

@njit(float64[:](float64[:,:,:],float64[:,:,:],int64[:],int64[:]), fastmath=True, parallel=True)
def closest_dist(coords1, coords2, bead_idx1, bead_idx2):
  
  p, n, dim = coords1.shape
  q, m, dim = coords2.shape
  pf = float(p)
  
  min_dists = np.empty(n)
  
  ncpu = 12
  
  for cpu in prange(ncpu):
    for i in range(cpu, n, ncpu):
      mean_inv_d2 = 0.0
      p1 = bead_idx1[i]
      dmin = -1
      
      for j in range(m):
        p2 = bead_idx2[j]
        
        if -3 < (p1-p2) < 3: # Close on backbone
          continue
         
        d2 = 0.0
 
        for k in range(p): # model
 
          dx = coords1[k,i,0] - coords2[k,j,0]
          dy = coords1[k,i,1] - coords2[k,j,1]
          dz = coords1[k,i,2] - coords2[k,j,2]

          d2 += max(1.0, dx * dx + dy * dy + dz * dz)
        
        d2 /= pf
        
        if dmin < 0:
          dmin = d2
        else:
          dmin = min(d2, dmin)
       
      min_dists[i] = np.sqrt(dmin)
   
  return min_dists


file_pairs = [('ESC','Cell1_anchor_ESC.bed','Cell1_P2E8_ambig_10x_100kb.n3d'),
              ('ESC','Cell2_anchor_ESC.bed','Cell2_P30E4_ambig_10x_100kb.n3d'),
              ('ESC','Cell4_anchor_ESC.bed','Cell3_Q6_ambig_10x_100kb.n3d'),
              ('ESC','Cell5_anchor_ESC.bed','Cell4_P2J8_ambig_10x_100kb.n3d'),
              ('ESC','Cell6_anchor_ESC.bed','Cell5_P2I5_ambig_10x_100kb.n3d'),
              ('ESC','P44F12_anchor_ESC.bed','P44F12_ambig_10x_100kb.n3d'),
              ('ESC','P44F6_anchor_ESC.bed','P44F6_ambig_10x_100kb.n3d'),
              ('ESC','P44H4_anchor_ESC.bed','P44H4_ambig_10x_100kb.n3d'),
              ('48h','P45F10_anchor_48h.bed','P45F10_ambig_10x_100kb.n3d'),
              ('48h','P46D12_anchor_48h.bed','P46D12_ambig_10x_100kb.n3d'),
              ('48h','P46D6_anchor_48h.bed','P46D6_ambig_10x_100kb.n3d'),
              ('48h','P46G10_anchor_48h.bed','P46G10_ambig_10x_100kb.n3d'),
              ('48h','P54E14_anchor_48h.bed','P54E14_ambig_10x_100kb.n3d'),
              ('48h','P54F7_anchor_48h.bed','P54F7_ambig_10x_100kb.n3d'),
              ('48h','P54G11_anchor_48h.bed','P54G11_ambig_10x_100kb.n3d'),
              ('48h','P54G12_anchor_48h.bed','P54G12_ambig_10x_100kb.n3d'),
              ('48h','P54G13_anchor_48h.bed','P54G13_ambig_10x_100kb.n3d'),
              ('48h','P54H12_anchor_48h.bed','P54H12_ambig_10x_100kb.n3d'),
              ('24h_H','P62E12_anchor_24h_H.bed','P62E12_ambig_10x_100kb.n3d'),
              ('24h_H','P62E6_anchor_24h_H.bed','P62E6_ambig_10x_100kb.n3d'),
              ('24h_H','P62F11_anchor_24h_H.bed','P62F11_ambig_10x_100kb.n3d'),
              ('24h_H','P62G8_anchor_24h_H.bed','P62G8_ambig_10x_100kb.n3d'),
              ('24h_H','P62H10_anchor_24h_H.bed','P62H10_ambig_10x_100kb.n3d'),
              ('24h_H','P62H13_anchor_24h_H.bed','P62H13_ambig_10x_100kb.n3d'),
              ('24h_L','P63E14_anchor_24h_L.bed','P63E14_ambig_10x_100kb.n3d'),
              ('24h_L','P63E9_anchor_24h_L.bed','P63G12_ambig_10x_100kb.n3d'),
              ('24h_L','P63G12_anchor_24h_L.bed','P63H10_ambig_10x_100kb.n3d'),
              ('24h_L','P63H14_anchor_24h_L.bed','P63H14_ambig_10x_100kb.n3d'),
              ('24h_L','P63H9_anchor_24h_L.bed','P63H9_ambig_10x_100kb.n3d'),
              ('24h_L','P64E11_anchor_24h_L.bed','P64E11_ambig_10x_100kb.n3d'),
              ('24h_L','P64E5_anchor_24h_L.bed','P64E5_ambig_10x_100kb.n3d'),
]

ab_paths = {'ESC':('/data/hi-c/transition_state/2i_SLX-7672_A.bed','/data/hi-c/transition_state/2i_SLX-7672_B.bed'),
            '24h_L':('/data/hi-c/transition_state/Rex1Low_merged_A.bed','/data/hi-c/transition_state/Rex1Low_merged_B.bed'),
            '24h_H':('/data/hi-c/transition_state/Rex1High_merged_A.bed','/data/hi-c/transition_state/Rex1High_merged_B.bed'),
            '48h':('/data/hi-c/transition_state/WT_merged_A.bed','/data/hi-c/transition_state/WT_merged_B.bed'),
            }
#
#file_pairs = [('ESC','Cell1_anchor_ESC.bed','Cell1_P2E8_ambig_10x_100kb.n3d'),
#              ('48h','P54H12_anchor_48h.bed','P54H12_ambig_10x_100kb.n3d'),
#              ('24h_H','P62E12_anchor_24h_H.bed','P62E12_ambig_10x_100kb.n3d'),
#              ('24h_L','P64E5_anchor_24h_L.bed','P64E5_ambig_10x_100kb.n3d'),]
#

ctypes = ['ESC','24h_H','24h_L','48h']

color_dict = {'ESC':'#000000','48h':'#FFB000','24h_H':'#0000FF','24h_L':'#00A0FF',}
label_dict = {'ESC':'Na\u00EFve ESC','48h':'Primed','24h_H':'24 h Rex1-high','24h_L':'24 h Formative',}
 
data_dir = '/data/hi-c/transition_state/hub_anchors/'

bg_dists = defaultdict(list)
hub_dists = defaultdict(list)
acomp_dists = defaultdict(list)
bcomp_dists = defaultdict(list)
x_min = 0.001
x_max = 0.009
n_bins = 20

for ctype, bed_file, n3d_file in file_pairs:
  print(ctype, bed_file, n3d_file)

  a_comp_path, b_comp_path = ab_paths[ctype]

  a_comp_data_dict = bed.load_data_track(a_comp_path)
  b_comp_data_dict = bed.load_data_track(b_comp_path)
  
  
  bed_path = data_dir + bed_file
  n3d_path = data_dir + n3d_file
  
  seq_pos_dict, coords_dict = n3d.load_n3d_coords(n3d_path)
  data_dict = bed.load_data_track(bed_path)

  chromos = util.sort_chromosomes([c for c in seq_pos_dict])
  
  file_bg_dists = []
  file_hub_dists = []
  file_a_dists = []
  file_b_dists = []
  
  for chromo in chromos:
    
    # Get trans dist distribs for all
    
    # Get trans dists for hubs in BED
    
    if chromo in data_dict:
      chrom_coords = coords_dict[chromo] 
      chrom_pos = seq_pos_dict[chromo]
      n_chrom_coords = chrom_coords.shape[1]
      
      pos1 = data_dict[chromo]['pos1']
      pos2 = data_dict[chromo]['pos2']
      
      seq_min = chrom_pos[0]
      seq_bin = chrom_pos[1] - seq_min
      seq_max = chrom_pos[-1]
      
      mid  = (pos1 + pos2)//2
      hub_idx = np.searchsorted(chrom_pos, mid)
      
      a_data = a_comp_data_dict[chromo]
      b_data = b_comp_data_dict[chromo]
      
      a_regions = np.stack([a_data['pos1'], a_data['pos2']], axis=1)
      b_regions = np.stack([b_data['pos1'], b_data['pos2']], axis=1)
      
      #Allocate large A/B regions to particles
      
      a_hist = util.bin_region_values(a_regions, np.ones(len(a_regions)), seq_bin, seq_min, seq_max)
      b_hist = util.bin_region_values(b_regions, np.ones(len(b_regions)), seq_bin, seq_min, seq_max)
      
      in_a = a_hist > 0.0
      in_b = b_hist > 0.0
      
      chr_idx = np.arange(n_chrom_coords)
      
      a_coords = chrom_coords[:,in_a]
      b_coords = chrom_coords[:,in_b]
      
      bead_coords = chrom_coords[:,hub_idx]
      other_coords = np.concatenate([coords_dict[x] for x in chromos if x != chromo], axis=1)
      
      # Just has to be outside current range
      trans_bead_idx = np.arange(other_coords.shape[1])+n_chrom_coords+10
      
      hub_cl_dists = dist_density(bead_coords, other_coords, hub_idx, trans_bead_idx)
      bg_cl_dists = dist_density(chrom_coords, other_coords, chr_idx, trans_bead_idx)
      
      a_cl_dists = dist_density(a_coords, other_coords, chr_idx[in_a], trans_bead_idx)
      b_cl_dists = dist_density(b_coords, other_coords, chr_idx[in_b], trans_bead_idx)

      file_bg_dists.append(bg_cl_dists)
      file_hub_dists.append(hub_cl_dists)
      file_a_dists.append(a_cl_dists)
      file_b_dists.append(b_cl_dists)
      #x_max = max(x_max, bg_cl_dists.max())
  
  file_bg_dists  = np.concatenate(file_bg_dists)
  file_hub_dists = np.concatenate(file_hub_dists)
  file_a_dists   = np.concatenate(file_a_dists)
  file_b_dists   = np.concatenate(file_b_dists)
  
  norm_scale = np.median(file_bg_dists)
  file_bg_dists /= norm_scale
  file_hub_dists /= norm_scale
  file_a_dists /= norm_scale
  file_b_dists /= norm_scale
  
  bg_dists[ctype].append(file_bg_dists)
  hub_dists[ctype].append(file_hub_dists)
  acomp_dists[ctype].append(file_a_dists)
  bcomp_dists[ctype].append(file_b_dists)

n_types = len(bg_dists)  
fig, axarr = plt.subplots(1, n_types+1)

plot_bg_dists_all = []
plot_hub_dists_all = []
plot_a_dists_all = []
plot_b_dists_all = []

xlabel = r'inter-chromo intermingle $\frac{1}{r^2}$ density'
ylabel = 'Probability density'

for i, ctype in enumerate(ctypes):
  plot_bg_dists =  np.concatenate(bg_dists[ctype])
  plot_hub_dists = np.concatenate(hub_dists[ctype])
  plot_a_dists =  np.concatenate(acomp_dists[ctype])
  plot_b_dists = np.concatenate(bcomp_dists[ctype])
  
  plot_bg_dists_all.append(plot_bg_dists)
  plot_hub_dists_all.append(plot_hub_dists)
  plot_a_dists_all.append(plot_a_dists)
  plot_b_dists_all.append(plot_b_dists)
  
  hist1, edges1 = np.histogram(plot_bg_dists, bins=24, density=True)
  hist2, edges2 = np.histogram(plot_hub_dists, bins=24, density=True)
  hist3, edges3 = np.histogram(plot_a_dists, bins=24, density=True)
  hist4, edges4 = np.histogram(plot_b_dists, bins=24, density=True)
   
  axarr[i].set_title(label_dict[ctype])
  axarr[i].plot(edges1[1:], hist1, color='#808080', alpha=0.3, linewidth=1, label='All sites')
  axarr[i].plot(edges3[1:], hist3, color='#004080', alpha=0.7, linestyle='--', linewidth=1, label='A comp')
  axarr[i].plot(edges4[1:], hist4, color='#800000', alpha=0.7, linestyle='--', linewidth=1, label='B comp')
  axarr[i].plot(edges2[1:], hist2, color=color_dict[ctype], alpha=0.7, linewidth=2, label='Hubs')

  axarr[i].legend(fontsize=9)
  axarr[i].set_xlabel(xlabel)
  axarr[i].set_ylabel(ylabel)
  #axarr[i].set_xlim(x_min, x_max)
  
i += 1

plot_bg_dists_all =  np.concatenate(plot_bg_dists_all)
plot_hub_dists_all = np.concatenate(plot_hub_dists_all)
plot_a_dists_all =  np.concatenate(plot_a_dists_all)
plot_b_dists_all = np.concatenate(plot_b_dists_all)

hist1, edges1 = np.histogram(plot_bg_dists_all, bins=n_bins, density=True)
hist2, edges2 = np.histogram(plot_hub_dists_all, bins=n_bins, density=True)
hist3, edges3 = np.histogram(plot_a_dists_all, bins=n_bins, density=True)
hist4, edges4 = np.histogram(plot_b_dists_all, bins=n_bins, density=True)
 
axarr[i].set_title('Aggregate')
axarr[i].plot(edges1[1:], hist1, color='#808080', alpha=0.3, linewidth=1, label='All sites')
axarr[i].plot(edges3[1:], hist3, color='#004080', alpha=0.7, linestyle='--', linewidth=1, label='A comp')
axarr[i].plot(edges4[1:], hist4, color='#800000', alpha=0.7, linestyle='--', linewidth=1, label='B comp')
axarr[i].plot(edges2[1:], hist2, color='#FF0000', alpha=0.7, linewidth=2, label='Hubs')

axarr[i].legend(fontsize=9)
axarr[i].set_xlabel(xlabel)
axarr[i].set_ylabel(ylabel)
#axarr[i].set_xlim(x_min, x_max)
  
plt.show()

# Should be more the depth below the trans surface

# Plot intermingling levels














