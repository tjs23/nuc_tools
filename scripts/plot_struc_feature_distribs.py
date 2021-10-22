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

plot_data = {}

for poi in poi_labels:
  plot_data[poi] = {}
  
  for sg in struc_labels:
    plot_data[poi][sg] = []


# Structure distances may need normalisation for better comparison

for s, s_group in enumerate(struc_groups):
  s_label = struc_labels[s]
  struc_files = glob(s_group)
  
  for struc_file in struc_files:
    print(f'Loading {struc_file}')
    seq_pos_dict, coords_dict = formats.n3d.load_n3d_coords(struc_file)
    chromos = sorted(seq_pos_dict)
    
    for chromo, seq_pos in seq_pos_dict.items():
      print(f' .. {chromo}')
      chromo_coords = coords_dict[chromo]
      
      a_data = a_comp_data_dict[chromo]
      b_data = b_comp_data_dict[chromo]
      
      # Need something better for below; these are big regions
      
      a_pos = (a_data['pos1'] + a_data['pos2']) // 2
      b_pos = (b_data['pos1'] + b_data['pos2']) // 2
      
      a_comp_coords = chromo_coords[p.searchsorted(seq_pos, a_pos)]
      b_comp_coords = chromo_coords[p.searchsorted(seq_pos, b_pos)]
      
      other_coords = np.concatenate([coords_dict[x] for x in chromos if x != chromo], axis=0)
      seq_min = seq_pos[0]
      seq_max = seq_pos[-1] + seq_pos[1] - seq_min
      
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
        
        plot_data[p_label][s_label].append(dists)
        
        # A/B boundary dists
        # get A/B allocation of pos
        # find closest dist in other comp
        
        
        
        
from matplotlib import pyplot as plt

fig, axarr = plt.subplots(len(plot_data), 1)

for row, p in enumerate(plot_data):
  ax = axarr[row]
  ax.set_title(p)
  ax.set_ylabel('Bin density')
  ax.set_xlabel('Structure distance')
  
  sub_dict = plot_data[p]
  
  for j, s in enumerate(sub_dict):
    vals = sub_dict[s]
    vals = np.concatenate(vals, axis=0)
    
    hist, edges = np.histogram(vals, bins=100)
    ax.plot(edges[:-1], hist, alpha=0.5, color=struc_colors[j], label=s)

plt.show()


