import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nuc_tools import util, io, parallel
from formats import bed, n3d
from numba import njit, float64, prange, int64
import multiprocessing

NCPU = multiprocessing.cpu_count()

@njit(float64[:](float64[:,:,:],float64[:,:,:],float64[:],int64[:],int64[:],int64,int64), fastmath=True, parallel=True)
def dist_density(coords1, coords2, weights2, idx1, idx2, ncpu, min_bead_sep):
  """
  coords1 : coords to calc densities at , e.g. points in one chromo
  coords2 : coords to use in the density calculations, typically whole structure
  weights2 : float values that apply at coords2 to scale densities
  idx1, idx2 : bead identity numbers for coord sets; bead number differences within min_bead_sep are skipped
  """
  m1, n1, dim = coords1.shape
  m2, n2, dim = coords2.shape
  # m1 == m2
  
  fm1 = float(m1)
  fn2 = float(n2)
  
  densities = np.empty(n1)

  for cpu in prange(ncpu):
    for i in range(cpu, n1, ncpu):
      wmean_inv_d2 = 0.0 # Mean weighted inverse square distance
      p1 = idx1[i]
      
      for j in range(n2):
        p2 = idx2[j]
        
        if -min_bead_sep < (p1-p2) < min_bead_sep: # Close on backbone
          continue
         
        d2 = 0.0
 
        for k in range(m1): # coordinate model
 
          dx = coords1[k,i,0] - coords2[k,j,0]
          dy = coords1[k,i,1] - coords2[k,j,1]
          dz = coords1[k,i,2] - coords2[k,j,2]

          d2 += max(1.0, dx * dx + dy * dy + dz * dz)
        
        
        # Average over models
        wmean_inv_d2 += weights2[j] * fm1/d2 # 1/mean(distance^2) = 1/(d2/fm1)
      
      # Average over target coords 
      densities[i] = wmean_inv_d2 / fn2
   
  return densities

# Paired inputs BED and 3D coords
file_pairs = [('/data/hi-c/transition_state/hub_anchors/Cell1_anchor_ESC.bed','/data/hi-c/transition_state/hub_anchors/Cell1_P2E8_ambig_10x_100kb.n3d'),
              #('Sample2.bed','Structure2.n3d'),
              #('Sample3.bed','Structure3.n3d'),
              ]

min_bead_sep = 3
strand = 1 # Arbitrary

for bed_path, n3d_path in file_pairs:
  print(f'Working on input pair {bed_path} : {n3d_path}')
  
  in_data_dict = bed.load_data_track(bed_path)
  seq_pos_dict, coords_dict = n3d.load_n3d_coords(n3d_path)
  chromos = util.sort_chromosomes([c for c in seq_pos_dict if c in in_data_dict]) # Only common to BED and N3D
  
  offset = 0
  nvals = 0
  bead_idx = {}
  binned_hist = {}
  for chromo in chromos:
    # set bead numbers (used to check seq adjacency), adding a gap between each chromosome
    bead_idx[chromo] = np.arange(offset, offset+len(seq_pos_dict[chromo]))
    offset += 2*min_bead_sep # arbitrary but big enough
    
    chrom_pos = seq_pos_dict[chromo] # Bead locations
    seq_min = chrom_pos[0]
    seq_max = chrom_pos[-1]
    seq_bin = chrom_pos[1] - seq_min
    
    bed_regions = np.stack([in_data_dict[chromo]['pos1'],
                            in_data_dict[chromo]['pos2']], axis=1)
    bed_values = in_data_dict[chromo]['value']
    
    # Bin BED data in bead regions for this chromo
    binned_hist[chromo] = util.bin_region_values(bed_regions, bed_values, seq_bin, seq_min, seq_max).astype(np.float64)    
    
  all_coords = np.concatenate([coords_dict[x] for x in chromos], axis=1) # Concat along atom/bead dim
  all_weights = np.concatenate([binned_hist[x] for x in chromos], axis=0)
  all_idx = np.concatenate([bead_idx[x] for x in chromos], axis=0)
  density_data_dict = {}
  dmax = 0.0
  dmin = float('inf')
  
  # Get densities
  dens_dict = {}
  
  for chromo in chromos:
    chrom_coords = coords_dict[chromo]
    chrom_idx = bead_idx[chromo] # Sequential bead numbers, different for each chromosome
    m, n, d = chrom_coords.shape
    print(f'  {chromo} m={m} n={n:,}')
    
    densities = dist_density(chrom_coords, all_coords, all_weights, chrom_idx, all_idx, NCPU, min_bead_sep)
    dens_dict[chromo] = densities
    
    nvals += n
    dmax = max(densities.max(), dmax)
    dmin = min(densities.min(), dmin)
  
  # Normalise, build output
  
  for chromo in chromos:
    chrom_pos = seq_pos_dict[chromo]
    seq_bin = chrom_pos[1] - chrom_pos[0]
    densities = (dens_dict[chromo] - dmin)/ (dmax-dmin) # Could normalise other ways...
    density_data_dict[chromo] = [(chrom_pos[i], chrom_pos[i]+seq_bin-1, strand, dens, dens, f'{i}') for i, dens in enumerate(densities)]
     
  out_bed_path = os.path.splitext(bed_path)[0] + '_density.bed'
  print(f'Save {out_bed_path} : {nvals:,} values')
  bed.save_data_track(out_bed_path, density_data_dict, scale=1.0, as_float=True)
