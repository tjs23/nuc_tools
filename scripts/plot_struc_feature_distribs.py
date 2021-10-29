import sys, os
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nuc_tools import util, io, parallel
from formats import bed, n3d


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


a_comp_data_dict = bed.load_data_track('/data/bed/A_comp.bed')
b_comp_data_dict = bed.load_data_track('/data/bed/B_comp.bed')

poi_paths = ['/data/hi-c/hybrid/chip-seq/K27me3_dec_then_inc.bed',
             '/data/hi-c/hybrid/chip-seq/K27me3_inc.bed',
             '/data/hi-c/hybrid/chip-seq/K27me3_dec.bed']

poi_labels = [f'Group {i+1}' for i in range(len(poi_paths))]

poi_dicts = [bed.load_data_track(x) for x in poi_paths]

struc_groups = [
                '/data/hi-c/transition_state/Haploid/EDL_ES_2iLIF/*_25k*.n3d',
                '/data/hi-c/transition_state/Haploid/EDL_24h_Rex1Low/*_25k*.n3d',
                '/data/hi-c/transition_state/Haploid/EDL_24h_Rex1High/*_25k*.n3d',
                '/data/hi-c/transition_state/Haploid/EDL_48h/*_25k*.n3d',
                ]
struc_groups = [
                '/data/hi-c/transition_state/Haploid/EDL_ES_2iLIF/*_100k*.n3d',
                '/data/hi-c/transition_state/Haploid/EDL_24h_Rex1Low/*_100k*.n3d',
                '/data/hi-c/transition_state/Haploid/EDL_24h_Rex1High/*_100k*.n3d',
                '/data/hi-c/transition_state/Haploid/EDL_48h/*_100k*.n3d',
                ]

struc_labels = ['0h','24h_Rex1Low','24h_Rex1High','48h']

struc_colors = ['#808080','#FF2000','#C0B000','#0080FF']

group_colors = ['#FF2000','#C0B000','#0080FF']

ic_plot_data = {}
a_plot_data = {}
b_plot_data = {}
st_plot_data = {}

for p_label in poi_labels:
  ic_plot_data[p_label] = {}
  a_plot_data[p_label] = {}
  b_plot_data[p_label] = {}
  st_plot_data[p_label] = {}
  
  for s_label in struc_labels:
    ic_plot_data[p_label][s_label] = {}
    a_plot_data[p_label][s_label] = {}
    b_plot_data[p_label][s_label] = {}

# Structure distances may need normalisation for better comparison
# Add to distance histograms on-the-fly to minimise memory


def _closest_dist(coords1, coords2, n_cpu=4):
  
  coords_list1 = np.array_split(coords1, n_cpu)
  
  results = parallel.run(_closest_dist, coords_list1, common_args=(coords2,),
                         num_cpu=len(coords_list1), verbose=False)
  
  closest_dists = np.concatenate(results, axis=0)
  
  return closest_dists
  
  
def closest_dist(coords1, coords2):
   
  dists2 = distance.cdist(coords1.mean(axis=0), coords2.mean(axis=0), 'sqeuclidean')
  
  z = dists2 == 0.0
  dists2[z] = 1.0
  dists2[dists2 < 1.0] = 1.0
  dists2 = 1.0/dists2
  dists2[z] = np.nan
  #return np.sqrt(dists2.min(axis=1))
  
  densities =  np.nanmean(dists2, axis=1)
  
  #print(coords1.shape, coords2.shape, densities.shape, densities.min(), densities.mean(), densities.max())
  
  return densities
 
 
#n_bins = 32
#hist_range = (0.0, 0.008)
n_bins = 24
hist_range = (0.0, 0.012)

def dist_hist(dists, n_bins=n_bins, hist_range=hist_range):
  
  hist, edges = np.histogram(dists[dists > 0.0], bins=n_bins, density=False, range=hist_range)
  
  return hist.astype(float)
  

for s, s_group in enumerate(struc_groups):
  s_label = struc_labels[s]
  struc_files = glob(s_group)
  
  for struc_file in struc_files:
    util.info(f'{struc_file}')
    
    seq_pos_dict, coords_dict = n3d.load_n3d_coords(struc_file)
    chromos = util.sort_chromosomes([c for c in seq_pos_dict if c in a_comp_data_dict])
    n_models = len(coords_dict[chromos[0]])
    
    """
    coords = np.concatenate([coords_dict[chromo] for chromo in chromos], axis=1)
    n_coords = coords.shape[1]
    dists = None
    for m in range(n_models):
      print(f' ..  {m}')
      if dists is None:
        dists = distance.pdist(coords[m], 'euclidean')
      else:
        dists += distance.pdist(coords[m], 'euclidean')
    
    dists /= n_models
    dists = dists[dists > 0.0]
    hist, edges = np.histogram(np.log10(dists), bins=25, density=True, range=(0.0, 1.0))
    st_plot_data[s_label][struc_file] = hist, edges
    """
          
    a_coords = []
    b_coords = []
    a_hists = {}
    b_hists = {}
    
    for chromo in chromos:
      seq_pos = seq_pos_dict[chromo]
      seq_min = seq_pos[0]
      seq_bin = seq_pos[1] - seq_min
      seq_max = seq_pos[-1]
       
      a_data = a_comp_data_dict[chromo]
      b_data = b_comp_data_dict[chromo]
      
      a_regions = np.stack([a_data['pos1'], a_data['pos2']], axis=1)
      b_regions = np.stack([b_data['pos1'], b_data['pos2']], axis=1)
      
      # Allocate large A/B regions to particles
      
      a_hist = util.bin_region_values(a_regions, np.ones(len(a_regions)), seq_bin, seq_min, seq_max)
      b_hist = util.bin_region_values(b_regions, np.ones(len(b_regions)), seq_bin, seq_min, seq_max)
      
      a_hists[chromo] = a_hist
      b_hists[chromo] = b_hist
      
      a_idx = a_hist.nonzero()[0]
      b_idx = b_hist.nonzero()[0]
      
      a_coords.append(coords_dict[chromo][:,a_idx])
      b_coords.append(coords_dict[chromo][:,b_idx])
 
    a_coords = np.concatenate(a_coords, axis=1)
    b_coords = np.concatenate(b_coords, axis=1)
        
    for chromo in chromos:
      util.info(f' .. {chromo}', line_return=True)
      seq_pos = seq_pos_dict[chromo]
      chromo_coords = coords_dict[chromo]
      
      seq_min = seq_pos[0]
      seq_bin = seq_pos[1] - seq_min
      seq_max = seq_pos[-1] + seq_bin
       
      other_coords = np.concatenate([coords_dict[x] for x in chromos if x != chromo], axis=1)
      background_dists = closest_dist(chromo_coords, other_coords)
      
      if struc_file in st_plot_data:
        st_plot_data[struc_file] += dist_hist(background_dists)
      else:
        st_plot_data[struc_file] = dist_hist(background_dists)
       
      for p, poi_dict in enumerate(poi_dicts):
        if chromo not in poi_dict:
          continue
      
        p_label = poi_labels[p]
        p_data = poi_dict[chromo]
        # For each point in this chromo get closest structure particle in a different chromo
        pos = (p_data['pos1'] + p_data['pos2']) // 2
        
        # Data must be within structure limits
        pos = pos[pos >= seq_min]
        pos = pos[pos < seq_max]
        
        bead_idx = np.clip(np.searchsorted(seq_pos, pos), 0, len(seq_pos)-1)
        bead_coords = chromo_coords[:,bead_idx]
        dists = closest_dist(bead_coords, other_coords)
        
        if struc_file in ic_plot_data[p_label][s_label]:
          ic_plot_data[p_label][s_label][struc_file] += dist_hist(dists)
        else:
          ic_plot_data[p_label][s_label][struc_file] = dist_hist(dists)
        
        # A/B boundary dists
        # get A/B allocation of pos
        
        #in_a = a_hists[chromo][bead_idx] > 0.0
        #in_b = b_hists[chromo][bead_idx] > 0.0
        #in_b[in_a & in_b] = False # Cannot be in both; at boundary anyhow
        
        # Find closest dist in the opposite compartment
        dists = closest_dist(bead_coords,  a_coords)       
        if struc_file in a_plot_data[p_label][s_label]:
          a_plot_data[p_label][s_label][struc_file] += dist_hist(dists)
        else:
          a_plot_data[p_label][s_label][struc_file] = dist_hist(dists)
        
        #dists = closest_dist(bead_coords, b_coords)
        #if struc_file in b_plot_data:
        #  b_plot_data[p_label][s_label][struc_file] += dist_hist(dists)
        #else:
        #  b_plot_data[p_label][s_label][struc_file] = dist_hist(dists)
        
        
from matplotlib import pyplot as plt

plot_data = [('Inter-chromosomal', ic_plot_data),
             ('A-compartment', a_plot_data),]
             #('B compartment', b_plot_data)]

fig, axarr = plt.subplots(len(struc_groups), len(plot_data), squeeze=False)
fig.set_size_inches(12.0, 12.0)
n_rows = len(struc_labels)

for col, (title, data_dict) in enumerate(plot_data):

  a, b = hist_range[0], hist_range[1]
  edges = np.arange(a, b, (b-a)/n_bins)
  
  for row, s_label in enumerate(struc_labels): # Each structure group
    ax = axarr[row,col]
    
    ax.set_title(s_label, color='#008080', loc='left')
    ax.set_ylabel('Bin density')
    
    if row == n_rows-1:
      ax.set_xlabel(f'{title} $1/r^2$ density')

    for j, p_label in enumerate(poi_labels):
      #exp_mean = np.zeros(n_bins)
      obs_mean = np.zeros(n_bins)
      
      for struc_file in data_dict[p_label][s_label]:
        #exp_hist = st_plot_data[struc_file]
        #exp_mean += exp_hist
        
        obs_hist = data_dict[p_label][s_label][struc_file]
        obs_mean += obs_hist
        

      obs_mean /= obs_mean.sum() or 1.0
      #ax.plot(edges, exp_mean, alpha=0.25, color=struc_colors[j], linestyle='--')
      ax.plot(edges, obs_mean, alpha=0.5, color=group_colors[j], label=p_label)
        
        
    if row == 0:    
      ax.legend()
    
plt.show()

fig, axarr = plt.subplots(len(poi_labels), len(plot_data), squeeze=False)
fig.set_size_inches(12.0, 12.0)

for col, (title, data_dict) in enumerate(plot_data):

  a, b = hist_range[0], hist_range[1]
  edges = np.arange(a, b, (b-a)/n_bins)
  
  for row, p_label in enumerate(data_dict): # Each gene group
    ax = axarr[row,col]
    
    ax.set_title(p_label, color='#008080', loc='left')
    ax.set_ylabel('Bin density')
    ax.set_xlabel(f'{title} $1/r^2$ density')
    #ax.set_ylim((-3.1, 2.1))
   
    sub_dict = data_dict[p_label]
       
    for j, s_label in enumerate(sub_dict):
      exp_mean = np.zeros(n_bins)
      obs_mean = np.zeros(n_bins)
      
      for struc_file in sub_dict[s_label]:
        exp_hist = st_plot_data[struc_file]
        #exp_hist /= exp_hist.max() or 1.0
        exp_mean += exp_hist
        
        obs_hist = sub_dict[s_label][struc_file]
        #obs_hist /= obs_hist.max() or 1.0
        obs_mean += obs_hist
        
        #nz = (exp_hist * obs_hist) != 0.0
        #mean_lr[nz] += np.log(obs_hist[nz]/exp_hist[nz])
        
      #exp_mean /= len(sub_dict)
      #obs_mean /= len(sub_dict)
      #exp_mean /= exp_mean.sum()
      obs_mean /= obs_mean.sum() or 1.0
      #ax.plot(edges, exp_mean, alpha=0.25, color=struc_colors[j], linestyle='--')
      ax.plot(edges, obs_mean, alpha=0.5, color=struc_colors[j], label=s_label)
        
    
plt.show()
