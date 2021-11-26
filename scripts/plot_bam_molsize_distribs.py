import sys, os
import numpy as np
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nuc_tools import io, util
from formats import sam

bam_paths = sys.argv[1:]

cmap = util.string_to_colormap('#B000B0,#FF4000,#00B0B0,#0000FF')

fig, (ax1, ax2) = plt.subplots(1,2)

n_paths = len(bam_paths)

histograms = []

labels = ['MNase 0.2', 'MNase 0.6', 'MNase 1.5 (B)', 'MNase 1.5 (T)', 'MNase 1.5 (B+T)']

for b, bam_path in enumerate(bam_paths):
  
  label = labels[b] # os.path.basename(bam_path)
  color = cmap(float(b)/(n_paths-1.0))
  
  sizes_pos = []
  sizes_pos_append = sizes_pos.append
  sizes_neg = []
  sizes_neg_append = sizes_neg.append
  util.info('Reading {}\n'.format(bam_path))
  
  hist_args = {'range':(100, 1500), 'bins':1400, 'density':True}
  
  for i, (rname, sam_flag, chromo, pos, mapq, cigar, mate_chromo, mate_pos, t_len, seq, qual) in enumerate(sam.bam_iterator(bam_path)):
    
    
    if i and i % 1000000 == 0:
      util.info(' .. {:,}'.format(i), line_return=True)
      
      #if i > 1.1e6:
      #  break
    
    
    if chromo == '*':
      continue
      
    #print sam_flag, chromo, pos, mapq, cigar, mate_chromo, mate_pos
    #raw_input('>>>')
    
    if mate_chromo != '=':
      continue
    
    size = len(seq) 
    
    sam_flag = int(sam_flag)
    
    if sam_flag & 0x4: # R1 unmapped
      continue

    if sam_flag & 0x8: # R2 unmapped
      continue

    if sam_flag & 0x100: # Secondary
      continue
      
    is_neg = sam_flag & 0x10
    pos_a = int(pos)
    pos_b = int(mate_pos)
    delta = abs(pos_a-pos_b) + size
    
    #delta = abs(int(t_len))
    
    #if is_neg: # Negative strand
    #  sizes_neg_append(delta)
    #else:
    
    sizes_pos.append(delta)
  
  util.info(' .. {:,}'.format(i), line_return=True)
  

  hist, edges = np.histogram(sizes_pos, **hist_args)
  ax1.plot(edges[:-1], hist, color=color, alpha=0.5, label=label, linewidth=1.0)
  
  histograms.append(hist)

  #hist, edges = np.histogram(sizes_neg, **hist_args)
  #ax.plot(edges[:-1], hist, color=color, alpha=0.3, linestyle='--', linewidth=0.3)

comb_hist = (histograms[-1] + histograms[-2]) * 0.5
ax2.plot(edges[:-1], histograms[0], color='#B000B0', alpha=0.5, label='MNase 0.2', linewidth=1.0)
ax2.plot(edges[:-1], histograms[1], color='#FF4000', alpha=0.5, label='MNase 0.6', linewidth=1.0)
ax2.plot(edges[:-1], comb_hist, color='#0080FF', alpha=0.5, label='MNase 1.5 (B+T)', linewidth=1.0)

ax1.set_xlabel('Read end separation (bp)')
ax1.set_ylabel('Count density (bp bins)')
ax2.set_xlabel('Read end separation (bp)')
ax2.set_ylabel('Count density (bp bins)')

ax1.legend()
ax2.legend()

plt.show()
