import sys, os, re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from nuc_tools import io, util

from formats import bed

ref_bed_path = sys.argv[1]

bed_paths = sys.argv[2:]


util.info('Loading {}'.format(ref_bed_path))

ref_data_dict = bed.load_data_track(ref_bed_path)

ref_max = float('-inf')
ref_min = float('inf')

bin_size = 1000

for chromo in ref_data_dict:
  
  vals = ref_data_dict[chromo]['value']
  
  ref_max = max(ref_max, max(vals))
  ref_min = min(ref_min, min(vals))
 
  
for bed_path in bed_paths:
  
  norm_bed_path = os.path.splitext(bed_path)[0] + '_norm.bed'
  
  util.info('Loading {}'.format(bed_path))
  
  data_dict = bed.load_data_track(bed_path)
   
  for chromo in data_dict:
    
    if chromo in ref_data_dict:
      
      dtype = data_dict[chromo].dtype
      
      print dtype
             
      vals = data_dict[chromo]['value']
      pos1 = data_dict[chromo]['pos1']
      pos2 = data_dict[chromo]['pos2']
      mids = (pos1 + pos2)//2
      
      ref_pos1 = ref_data_dict[chromo]['pos1']
      ref_pos2 = ref_data_dict[chromo]['pos2']
      ref_mids = (ref_pos1 + ref_pos2)//2
      
      ref_vals =  ref_data_dict[chromo]['value']
      ref_vals -= ref_min
      ref_vals /= ref_max
      
      pos_min = min(pos1[0], ref_pos1[0])
      pos_max = max(pos1[-1], ref_pos1[-1])
      
      edges = np.arange(pos_min, pos_max+bin_size+1, bin_size)
      hist = np.zeros(len(edges))
      ref_hist = np.zeros(len(edges))
      
      idx = np.searchsorted(edges, mids)
      hist[idx] += vals
      
      idx = np.searchsorted(edges, ref_mids)
      ref_hist[idx] += ref_vals
      ref_hist[ref_hist == 0.0] = 1.0
      
      hist /= ref_hist
      
      valid = hist != 0.0
      vals = hist[valid]
      pos1 = edges[valid]
      pos2 = pos1 + (bin_size-1)
      n = len(vals)
      strands = np.ones(n, dtype=bool)
      
      data_dict[chromo] = np.empty(n, dtype=dtype)
      
      data_dict[chromo]['orig_value'] = vals
      data_dict[chromo]['value'] = vals
      data_dict[chromo]['pos1'] = pos1
      data_dict[chromo]['pos2'] = pos2
      data_dict[chromo]['strand'] = np.ones(n, dtype=bool)
      data_dict[chromo]['label'] = [str(i) for i in range(n)]

  
  util.info('Saving {}'.format(norm_bed_path))
  
  bed.save_data_track(norm_bed_path, data_dict, as_float=True)
