import sys, os, time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict
from nuc_tools import util, io
from tools.contact_map import get_single_list_matrix
from formats import bed, ncc

from matplotlib import pyplot as plt
from scipy import signal

# Mini gene detector; from GFF in, GFF out, purely on basis of size, which is consistent

# Based on population Hi-C, a delineation of domains; start, end, inter-domain regions, all edges

# Input is NCC; good resolution important

# Must be of a given mimimum size

# Cannot use flips, given IDRs

# For a given bin size (smoothed at higher res) find a before/after relative step in mid-range contacts; 
# maximize forward step size in local region; step could be a ratio
# plot step size distribution to optimise thresholds
# the separation extent is the estimate for the other end; refine other end based on max backward step
# can test half the region, extent defined based on a high/low count split; half way between min and max off-diagonal
# gives end test extent +/- error 

# Should condier ambigous mappings

def write_bed_regions(bed_file_path, chr_region_dict):

  data_dict = {}
  for chromo in chr_region_dict:
    regions = chr_region_dict[chromo]
    
    if len(regions):
      value_anno = np.ones(regions.shape)
      value_anno[:,2] = 0
 
      data_dict[chromo] = np.concatenate([regions, value_anno], axis=1)

  bed.save_data_track(bed_file_path, data_dict)
  
  
def ncc_domain_detect(ncc_path, bin_size=int(1e3), region_bins=200, min_chr_size=1000e3, plot=False):
  
  file_root, file_ext = os.path.splitext(ncc_path)
  while file_ext.lower() in ('.gz', '.gzip'):
    file_root, file_ext = os.path.splitext(file_root)
  
  size = int(bin_size * region_bins * 1e-3)
  
  bed_file_path_scr = f'{file_root}_edge_scores_{size}k.bed'
  bed_file_path_reg = f'{file_root}_dom_regions_{size}k.bed'
  
  util.info('Loading NCC format contact data')
  chromosomes, chromo_limits, contacts = ncc.load_file(ncc_path)
  
  ambig_groups = defaultdict(int)
    
  for key in contacts:
    for p_a, p_b, nobs, ag in contacts[key]:
      ambig_groups[ag] += 1 
  
  data_dict_scr = defaultdict(list)
  data_dict_reg = defaultdict(list)
  
  
  for chr_a, chr_b in contacts:
    if chr_a != chr_b:
      continue
    
    limits = chromo_limits[chr_a]
    
    if limits[1]-limits[0] < min_chr_size:
      continue
    
    matrix, ambig_matrix = get_single_list_matrix(contacts[(chr_a, chr_a)], limits, limits,
                                                  True, bin_size, ambig_groups, smooth=False)

    n, m = matrix.shape
 
    util.info(f'{chr_a} : {n:,} bins')
    
    xx, yy =  np.mgrid[0:n,0:n]
    zz = np.abs(xx-yy).astype(float)
    matrix2 = matrix * zz # Scale with the square of the idstance to diagonal
    
    fwd_bias = np.zeros(n)
    bwd_bias = np.zeros(n)
    
    for i in range(10, n):
      a = max(0, i-region_bins)
      b = min(i+region_bins, n)
   
      top_rect = matrix2[a:i,i:b].sum()
      forward  = matrix2[i:b,i:b].sum()
      backward = matrix2[a:i,a:i].sum()
      
      area_bwd = (i-a) * (i-a)
      area_fwd = (b-i) * (b-i)
      area_top = (b-i) * (i-a)
      
      top_rect/= area_top
      forward /= area_fwd
      backward /= area_bwd
      
      if forward and top_rect:
        fwd_bias[i] = forward - top_rect
      
      if forward and top_rect:
        bwd_bias[i] = backward - top_rect
    
    #(start, end, strand, value, orig_value, label) in enumerate(data_dict[chromo]
    for i, val in enumerate(fwd_bias):
      if val > 0.1:
        val -= 0.09
        pos = int(i*bin_size)
        data_dict_scr[chr_a].append((pos, pos+bin_size, 1, val*100, val*100, ''))

    for i, val in enumerate(bwd_bias):
      if val > 0.1:
        val -= 0.09
        pos = int(i*bin_size)
        data_dict_scr[chr_a].append((pos, pos+bin_size, 0, val*100, val*100, ''))
    
    fwd_bias = signal.savgol_filter(fwd_bias, 41, 3)
    bwd_bias = signal.savgol_filter(bwd_bias, 41, 3)
        
    min_height = 0.25
    pos_peak_idx, pos_peak_props = signal.find_peaks(fwd_bias, height=min_height, distance=region_bins//2)
    neg_peak_idx, neg_peak_props = signal.find_peaks(bwd_bias, height=min_height, distance=region_bins//2)
    
    x_vals = np.arange(0, n)    
    
    if plot:
      fig, ax = plt.subplots()
      ax.set_title(chr_a)
      ax.plot(x_vals, fwd_bias, alpha=0.5, color='#0080FF')
      ax.plot(x_vals, bwd_bias, alpha=0.5, color='#FF0000')
 
      ax.scatter(x_vals[pos_peak_idx], fwd_bias[pos_peak_idx], color='#004080', s=20)
      ax.scatter(x_vals[neg_peak_idx], bwd_bias[neg_peak_idx], color='#800000', s=20)
      
    # for each pos signal get approx location of negative and find refined negative location;     
    # repeat for negatives if not already found
    w = region_bins//2
    
    domain_regions = []
    neg_peak_idx = set(neg_peak_idx)
    pos_peak_idx = set(pos_peak_idx)
    
    for i in pos_peak_idx:
      b = min(i+region_bins, n)
      profile = matrix[i:b,i:].sum(axis=0)
      thresh = max(3.0, 0.5 * np.median(profile[profile>0.0]))
      k = (profile > thresh).nonzero()[0][-1]
      j = i + k
      r1 = max((i+j)//2, j-w)
      r2 = min(n, j+w)
      
      for k in range(r1, r2):
        if k in neg_peak_idx:
          domain_regions.append((i,k))
          neg_peak_idx.remove(k)
          break
      else:
        k = r1 + bwd_bias[r1:r2].argmax()
        domain_regions.append((i,k))
      
    for i in neg_peak_idx:
      a = max(0, i-region_bins)
      profile = matrix[a:i,:i].sum(axis=0)
      thresh = max(3.0, 0.5 * np.median(profile[profile>0.0]))
      k = len(profile) - (profile > thresh).nonzero()[0][0]
      j = i - k
      
      r1 = max(0, j-w)
      r2 = min((i+j)//2, j+w)
      k = r1 + fwd_bias[r1:r2].argmax()
      domain_regions.append((k,i))
    
    for a, b in domain_regions:
      if plot:
        ax.plot([a,b],[fwd_bias[a], bwd_bias[b]], color='#808080')
      
      pos1 = int(a*bin_size)
      pos2 = int(b*bin_size)
      data_dict_reg[chr_a].append((pos1, pos2, '+', 1.0, 1.0, ''))
       
    if plot:        
      plt.show()
  
  for chromo in data_dict_scr:
    data_dict_scr[chromo].sort()

  for chromo in data_dict_reg:
    data_dict_reg[chromo].sort()
  
  bed.save_data_track(bed_file_path_scr, data_dict_scr)
  bed.save_data_track(bed_file_path_reg, data_dict_reg)
    
    
    
ncc_path = '/data/dino_hi-c/SLX-17943_HEM_5_HiC_cis.ncc.gz'

ncc_domain_detect(ncc_path)
