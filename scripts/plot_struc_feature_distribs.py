import sys, os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nuc_tools import util, io, formats


"""
For EDL

Pictures of typical cells at three time point (0, 24, 48 hr) with three different gene classes, highlighted vs trans distance
[9 images]

Three gene sets

Plot distribution structural values like trans distance, A depth, A/B boundary dist
 for the three gene sets 


tool to take a number of structures 


"""

from scipy.spatial import distance

a_comp_data_dict = formats.bed.load_data_track('')
b_comp_data_dict = formats.bed.load_data_track('')

poi_paths = ['',
             '',
             '']

poi_labels = [f'Group {i+1}' for i in range(len(poi_paths))]

poi_dicts = [formats.bed.load_data_track(x) for x in poi_paths]

struc_groups = [
                '/data/hi-c/hybrid/transition_state/Haploid/EDL_ES_2iLIF/*._10x_100kb.n3d',
                '/data/hi-c/hybrid/transition_state/Haploid/EDL_24h_Rex1Low /*._10x_100kb.n3d',
                '/data/hi-c/hybrid/transition_state/Haploid/EDL_24h_Rex1High/*._10x_100kb.n3d',
                '/data/hi-c/hybrid/transition_state/Haploid/EDL_48h/*._10x_100kb.n3d',
                ]

struc_labels = ['0h','24h_Rex1Low','24h_Rex1High','48h']

struc_colors = ['#808080','#FF2000','#C0B000','#0080FF']

ic_plot_data = {}
ab_plot_data = {}

for poi in poi_labels:
  ic_plot_data[poi] = {}
  ab_plot_data[poi] = {}
  
  for sg in struc_labels:
    ic_plot_data[poi][sg] = []
    ab_plot_data[poi][sg] = []

# Structure distances may need normalisation for better comparison
# Add to distance histograms on-the-fly to minimise memory


for s, s_group in enumerate(struc_groups):
  s_label = struc_labels[s]
  struc_files = glob(s_group)
  
  for struc_file in struc_files:
    print(f'Loading {struc_file}')
    seq_pos_dict, coords_dict = formats.n3d.load_n3d_coords(struc_file)
    chromos = sorted(seq_pos_dict)
    
    a_idx_dict = {}
    b_idx_dict = {}
    
    for chromo, seq_pos in seq_pos_dict.items():
      seq_min = seq_pos[0]
      seq_bin = seq_pos[1] - seq_min
      seq_max = seq_pos[-1] + seq_bin
       
      a_data = a_comp_data_dict[chromo]
      b_data = b_comp_data_dict[chromo]
      
      a_regions = np.stack([a_data['pos1'], a_data['pos2']], axis=1)
      b_regions = np.stack([b_data['pos1'], b_data['pos2']], axis=1)
      
      # Allocate large A/B regions to particles
      
      a_hist = util.bin_region_values(a_regions, np.ones(a_regions.shape), seq_bin, seq_min, seq_max)
      b_hist = util.bin_region_values(b_regions, np.ones(b_regions.shape), seq_bin, seq_min, seq_max)

      a_idx_dict[chromo] = a_hist.nonzero()[0]
      b_idx_dict[chromo] = b_hist.nonzero()[0]
 
    a_coords = np.concatenate([a_idx_dict[chromo] for chromo in chromos], axis=0)
    b_coords = np.concatenate([b_idx_dict[chromo] for chromo in chromos], axis=0)
    
    for chromo, seq_pos in seq_pos_dict.items():
      print(f' .. {chromo}')
      chromo_coords = coords_dict[chromo]
      
      seq_min = seq_pos[0]
      seq_bin = seq_pos[1] - seq_min
      seq_max = seq_pos[-1] + seq_bin
       
      other_coords = np.concatenate([coords_dict[x] for x in chromos if x != chromo], axis=0)
      
      for p, poi_dict in enumerate(poi_dicts):
        p_label = poi_labels[p]
        p_data = poi_dict[chromo]
        # For each point in this chromo get closest structure particle in a different chromo
        pos = (p_data['pos1'] + p_data['pos2']) // 2
        
        # Data must be within structure limits
        pos = pos[pos >= seq_min]
        pos = pos[pos < seq_max]
        
        bead_idx = np.searchsorted(seq_pos, pos)
        bead_coords = chromo_coords[bead_idx]
        
        dists2 = distance.cdist(bead_coords, other_coords, 'sqeuclidean')
        closest = dists2.argmin(axis=1)
        dists = np.sqrt(dists2[:,closest])
        
        ic_plot_data[p_label][s_label].append(dists)
        
        # A/B boundary dists
        # get A/B allocation of pos
        
        in_a = a_hist[bead_idx] > 0.0
        in_b = b_hist[bead_idx] > 0.0
        in_b[in_a & in_b] = False # Cannot be in both; at boundary anyhow
        
        # Find closest dist in the opposite compartment
        dists2 = distance.cdist(bead_coords[in_a], b_coords, 'sqeuclidean')
        closest = dists2.argmin(axis=1)
        dists = np.sqrt(dists2[:,closest])
        ab_plot_data[p_label][s_label].append(dists)       
        
        dists2 = distance.cdist(bead_coords[in_b], a_coords, 'sqeuclidean')
        closest = dists2.argmin(axis=1)
        dists = np.sqrt(dists2[:,closest])
        ab_plot_data[p_label][s_label].append(dists)       
        
        
        
        
from matplotlib import pyplot as plt

for title, plot_data in [('Trans chromosome', ic_plot_data),
                         ('A/B boundary', ab_plot_data)]:

  fig, axarr = plt.subplots(len(plot_data), 1)

  for row, p in enumerate(plot_data):
    ax = axarr[row]
    ax.set_title(p)
    ax.set_ylabel('Bin density')
    ax.set_xlabel('f{title} distance')
 
    sub_dict = plot_data[p]
 
    for j, s in enumerate(sub_dict):
      vals = sub_dict[s]
      vals = np.concatenate(vals, axis=0)
 
      hist, edges = np.histogram(vals, bins=100)
      ax.plot(edges[:-1], hist, alpha=0.5, color=struc_colors[j], label=s)

  plt.show()


