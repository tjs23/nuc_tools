import sys, os, math, time
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nuc_tools import io, util
from formats import sam
from collections import defaultdict

from umap import UMAP

def extract_mol_size_vectors(bam_paths, bin_size, size_step=50, min_size=150, max_size=750):
  
  min_chromo_size = 300e3
  t0 = time.time()
  
  chromo_sizes = defaultdict(int)
  for bam_path in bam_paths:
    for chromo, size in sam.get_bam_chromo_sizes(bam_path):
      chromo_sizes[chromo] = max(chromo_sizes[chromo], size)  
  
  chromo_hists = {}
  n_feats = (max_size-min_size)//size_step
  
  for chromo, size in chromo_sizes.items():
    if chromo_sizes[chromo] < min_chromo_size:
      continue
    n_bins = int(math.ceil(size / bin_size))
    chromo_hists[chromo] = np.zeros((n_bins, n_feats))
    
  for bam_path in bam_paths:
    
    for i, (rname, sam_flag, chromo, pos, mapq, cigar, mate_chromo, mate_pos, t_len, seq, qual) in enumerate(sam.bam_iterator(bam_path)):
    
      if i % 100000 == 0:
        util.info('  .. {:,} {:7.2f}s'.format(i, time.time()-t0), line_return=True)
 
      #if i > 1e7:
      #  break
 
      if chromo == '*':
        continue
 
      if mate_chromo != '=':
        continue
 
      if chromo_sizes[chromo] < min_chromo_size:
        continue
 
      sam_flag = int(sam_flag)
 
      if sam_flag & 0x4: # R1 unmapped
        continue

      if sam_flag & 0x8: # R2 unmapped
        continue

      if sam_flag & 0x100: # Secondary
        continue
 
      p1 = int(pos)
      p2 = int(mate_pos)
 
      if p1 > p2:
        p1, p2, = p2, p1
 
      size = int(t_len) # len(seq) + p2 - p1
      if size >= max_size:
        continue
 
      if size < min_size:
        continue
 
      p2 = p1 + size
      p0 = (p1 + p2) // 2
      
      j = p0 // bin_size
      k = (size-min_size) // size_step
      
      chromo_hists[chromo][j,k] += 1
    
  data = np.concatenate([chromo_hists[chromo] for chromo in chromo_hists], axis=0)
  
  col_means = np.mean(data, axis=0)[None,:]
  col_std = np.std(data, axis=0)[None,:]
  
  data -= col_means
  data /= col_std
  
   
  return np.array(data)


if __name__ == '__main__':
  
  bam_paths = ['/data/dino_hi-c/SLX-17948_waller_hem_Mnase-02_sf.bam',
               '/data/dino_hi-c/SLX-17948_waller_hem_Mnase-06_sf.bam',
               '/data/dino_hi-c/SLX-17948_waller_hem_Mnase-15_sf.bam']
  
  data = extract_mol_size_vectors(bam_paths, bin_size=500, size_step=50, min_size=150, max_size=750)
  
  n, d = data.shape
  
  util.info(f'Extracted {n:,} vectors of size {d}')
  
  fig_size = 8.0
  
  umap = UMAP(n_neighbors=16, min_dist=0.1, n_components=2, metric='correlation').fit_transform(data)
  
  x_vals, y_vals = umap.T
  
  fig = plt.figure()
  fig.set_size_inches(fig_size, fig_size)
 
  ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # [left, bottom, width, height] 
  ax1.set_title('MNase vector UMAP')

  ax1.scatter(x_vals, y_vals, color='#0080FF', alpha=0.15, s=2)
   
  plt.show()
  
  
