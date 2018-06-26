import numpy as np

from matplotlib import pyplot as plt

from collections import defaultdict

from nuc_tools import util, io

ncc_path = '/data/hi-c/pop_HybridES0418.ncc.gz'
#ncc_path = '/data/hi-c/pop_HybridES0418_unambig.ncc'
#ncc_path = '/data/hi-c/pop_HybridES0418_chr1.ncc'

chromo_names_path_1 = '/data/genome/mm_129_chr_names.tsv'
chromo_names_path_2 = '/data/genome/mm_CAST_chr_names.tsv'
align_coords_path = '/data/genome/mm_129-CAST_align.coords.gz'

val_max = 2.1e8
bin_size = 1e4
n_bins = int(val_max/bin_size)

g1_counts = np.ones(n_bins, float)
g2_counts = np.ones(n_bins, float)
c_counts = np.zeros(n_bins, float)

chromo_contigs_1 = {}
chromo_contigs_2 = {}
contig_chromos_1 = {}
contig_chromos_2 = {}

util.info('Reading chromosome names')

with io.open_file(chromo_names_path_1) as file_obj:
  for line in file_obj:
    contig, chromo = line.split()
    chromo_contigs_1[chromo] = contig
    contig_chromos_1[contig] = chromo

with io.open_file(chromo_names_path_2) as file_obj:
  for line in file_obj:
    contig, chromo = line.split()
    chromo_contigs_2[chromo] = contig
    contig_chromos_2[contig] = chromo

hc_dict = {}
for chromo in chromo_contigs_1:
  contig_1 = chromo_contigs_1[chromo]
  contig_2 = chromo_contigs_2.get(chromo)
  
  if contig_1 and contig_2:
    hc_dict[contig_1] = contig_2
    hc_dict[contig_2] = contig_1

util.info('Reading genome alignment coords')

equiv_regions = defaultdict(list)

with io.open_file(align_coords_path) as file_obj:
  line = file_obj.readline()
  
  while line[:5] != '=====':
    line = file_obj.readline()  
  
  for line in file_obj:
    data = line.split()
    contig1, contig2 = data[-2:]
    
    if hc_dict.get(contig1) == contig2:
      chr1 = contig_chromos_1.get(contig1, contig1)
      chr2 = contig_chromos_2.get(contig2, contig2)
      s1, e1 = data[0:2]
      s2, e2 = data[3:5]
      s1 = int(s1)
      e1 = int(e1)
      e2 = int(e2)
      s2 = int(s2)
      
      #equiv_regions1[chr1].append((s1, e1, s2, e2))
      equiv_regions[chr2].append((s2, e2, s1, e1))

equiv_ends = {}

for chromo, regions in equiv_regions.iteritems():
  sort_list = sorted(regions)
  equiv_regions[chromo] = sort_list
  equiv_ends[chromo] = np.array([x[1] for x in sort_list])

from numpy import searchsorted

util.info('Reading contact mapping ambiguity')
    
ambig_counts = defaultdict(int)
with io.open_file(ncc_path) as in_file_obj:
  k = 0
  for line in in_file_obj:
    k += 1
    
    if k % 100000 == 0:
      util.info('  .. {:,}'.format(k))
      
    ambig_counts[line.split()[12]] += 1 
    

# Genomic positions need to be aligned

# Take one genome map it to the other and detect where equivalent, mappable positions are

# Read mummer coorinates output

# For each chromosome get mapping to homologous chromosome

# Should map to both HCs, unambiguously

util.info('Processing contacts')
cc = 0

with io.open_file(ncc_path) as file_obj:
  k = 0
    
  for line in file_obj:
    k += 1
    
    if k % 100000 == 0:
      util.info('  .. {:,}'.format(k))
        
    chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, \
      chr_b, start_b, end_b, f_start_b, f_end_b, strand_b, \
      ambig_group, pair_id, swap_pair = line.split()
    
    if ambig_counts[ambig_group] > 1:
      continue
    
    if chr_a == chr_b:
      continue

    if 'chr2' not in chr_a:
      continue
      
    if 'chr2' not in chr_b:
      continue

    if '.' not in chr_a:
      continue
      
    if '.' not in chr_b:
      continue
    
    root_a, gen_a = chr_a.split('.')
    root_b, gen_b = chr_b.split('.')    
    
    if root_a != root_b:
      continue
    
    pos_a = int(f_start_a) if strand_a == '+' else  int(f_end_a)
    pos_b = int(f_start_b) if strand_b == '+' else  int(f_end_b)
    
    if gen_a == 'a':
      i = int(pos_a/bin_size)
      g1_counts[i] += 1
    else:
      r = searchsorted(equiv_ends[root_a], pos_a)
      r = max(0, min(len(equiv_ends[root_a])-1, r))
      s2, e2, s1, e1 = equiv_regions[root_a][r]
      d2 = e2-pos_a
      pos_a = e1-d2   
      i = int(pos_a/bin_size)
      g2_counts[i] += 1 
 
    if gen_b == 'a':
      j = int(pos_b/bin_size)
      g1_counts[j] += 1
      
    else:
      r = searchsorted(equiv_ends[root_b], pos_b)
      r = max(0, min(len(equiv_ends[root_b])-1, r))
      s2, e2, s1, e1 = equiv_regions[root_b][r]
      d2 = e2-pos_b
      pos_b = e1-d2
      j = int(pos_b/bin_size)
      g2_counts[j] += 1
    
    if abs(pos_a-pos_b) < bin_size:
      cc += 1
      c_counts[i] += 1
      c_counts[j] += 1

print cc

fig, (ax1, ax2) = plt.subplots(2,1)
alpha = 0.5
thresh = 1e-4

p1_counts = g1_counts / g1_counts.sum()
p2_counts = g2_counts / g2_counts.sum()

h1 = -p1_counts * np.log2(p1_counts/p2_counts)
h2 = -p2_counts * np.log2(p2_counts/p1_counts)

#idx_u = (h1 > thresh).nonzero()[0]
#idx_l = (h2 > thresh).nonzero()[0]

idx_u = (c_counts > 3).nonzero()[0]
idx_l = c_counts.nonzero()[0]

idx = set(range(len(p1_counts)))
idx -= set(idx_u)
idx -= set(idx_l)
idx = sorted(idx)

ax1.scatter(g1_counts[idx], g2_counts[idx], color='#B0B000', alpha=alpha)
ax1.scatter(g1_counts[idx_l]-0.33, g2_counts[idx_l], color='#0080FF', alpha=alpha)
ax1.scatter(g1_counts[idx_u]+0.33, g2_counts[idx_u], color='#FF0000', alpha=alpha)
ax1.set_xlabel('Genome A')
ax1.set_ylabel('Genome B')
#ax1.plot(g1_counts, color='#0080FF', alpha=alpha)
#ax1.plot(-g2_counts, color='#FF4000', alpha=alpha)
#ax1.plot(c_counts, color='#000000', alpha=alpha)

#d_counts = np.array([g1_counts, g2_counts]).T
#d_counts = d_counts.max(axis=1) ** 2
#d_counts = np.clip(1, d_counts.max(), d_counts)

g1_counts += 1.0
g2_counts += 1.0

d_counts = h1 + h2

ax2.scatter(c_counts, d_counts, color='#0080FF', alpha=alpha)
ax2.set_xlabel('Unambig inter-homolog contacts')
ax2.set_ylabel('Hybrid contact asymmetry')

#ax2.plot(d_counts, color='#0080FF', alpha=alpha)
#ax2.plot(c_counts, color='#000000', alpha=alpha)

plt.show()
