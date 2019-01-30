import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

ncc_path = '/data/hi-c/pop_HybridES0418_chr1.ncc'
ncc_path = '/data/hi-c/pop_HybridES0418_001.ncc.gz'
ncc_path = '/home/tjs23/Desktop/proc_fix/diploid_3_145_test7.ncc'

import numpy as np

from collections import defaultdict

from nuc_tools import util, io

from formats import fasta

from matplotlib import pyplot as plt

from numpy import searchsorted

sys.path.append('/home/tjs23/ihs/')

from NewBlast import BlastSearch

from Homology import multiAlign

def rev_comp(sequence, isDna=True):
    
  from string import maketrans

  if isDna:
    sequence = sequence.replace('U','T')
    transTable = maketrans('ATGC', 'TACG')

  else:
    sequence = sequence.replace('T','U')
    transTable = maketrans('AUGC', 'UACG')

  complement = sequence.translate(transTable)
  reverseComp = complement[::-1]

  return reverseComp  
  
genome_path_1 = '/data/genome/GCA_001624185.1_129S1_SvImJ_v1_genomic.fna'
genome_path_2 = '/data/genome/GCA_001624445.1_CAST_EiJ_v1_genomic.fna'

blastdb_path_1 = '/data/genome/mm_129_db'
blastdb_path_2 = '/data/genome/mm_CAST_db'

blastdb_r1 = '/data/hi-c/SLX-15555/SLX-15555_r1_bastdb'
blastdb_r2 = '/data/hi-c/SLX-15555/SLX-15555_r2_bastdb'

chromo_names_path_1 = '/data/genome/mm_129_chr_names.tsv'
chromo_names_path_2 = '/data/genome/mm_CAST_chr_names.tsv'

close_cis = 1e5

util.info('Reading chromo names')

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
    
util.info('Reading genome build FASTAs')
seq_dict_1 = fasta.read_fasta(io.open_file(genome_path_1)) # , max_seqs=1)
seq_dict_2 = fasta.read_fasta(io.open_file(genome_path_2)) # , max_seqs=1)

util.info('Reading ambiguity groups')

ambig_counts = defaultdict(int)
with io.open_file(ncc_path) as in_file_obj:
  for line in in_file_obj:
    ambig_counts[line.split()[12]] += 1      


util.info('Reading genome alignment coords')

align_coords_path = '/data/genome/mm_129-CAST_align.coords.gz'
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

    
util.info('Analysis')

fig, ax1 = plt.subplots()

x_vals = []
y_vals = []
cc = 0

with io.open_file(ncc_path) as file_obj:
  k = 0
    
  for line in file_obj:
    k += 1
    
    #if k % 100000 == 0:
    #  util.info('  .. {:,}'.format(k))
  
    chr_a, start_a, end_a, f_start_a, f_end_a, strand_a, \
      chr_b, start_b, end_b, f_start_b, f_end_b, strand_b, \
      ambig_group, pair_id, swap_pair = line.split()
    
    if ambig_counts[ambig_group] > 1:
      continue
    
    if chr_a == chr_b:
      continue

    if 'chr4.' not in chr_a:
      continue
      
    if 'chr4.' not in chr_b:
      continue

    if '.' not in chr_a:
      continue
      
    if '.' not in chr_b:
      continue
    
    root_a, gen_a = chr_a.split('.')
    root_b, gen_b = chr_b.split('.')    
    
    if root_a != root_b:
      continue
    
    start_a = int(start_a)
    end_a = int(end_a)
    start_b = int(start_b)
    end_b = int(end_b)
    
    if start_a > end_a:
      start_a, end_a = end_a, start_a
      
    if start_b > end_b:
      start_b, end_b = end_b, start_b
    
    pos_a = int(f_start_a) if strand_a == '+' else  int(f_end_a)
    pos_b = int(f_start_b) if strand_b == '+' else  int(f_end_b)

    if gen_a == 'a': # First genotype
      contig_a = chromo_contigs_1[root_a]
      seq_dict_a = seq_dict_1
      blastdb_path_a = blastdb_path_1
    else:
      contig_a = chromo_contigs_2[root_a]
      seq_dict_a = seq_dict_2
      blastdb_path_a = blastdb_path_2
      
      r = searchsorted(equiv_ends[root_a], pos_a)
      r = max(0, min(len(equiv_ends[root_a])-1, r))
      s2, e2, s1, e1 = equiv_regions[root_a][r]
      d2 = e2-pos_a
      pos_a = e1-d2      

    if gen_b == 'a':
      contig_b = chromo_contigs_1[root_b]
      seq_dict_b = seq_dict_1
      blastdb_path_b = blastdb_path_1
    else:
      contig_b = chromo_contigs_2[root_b]
      seq_dict_b = seq_dict_2
      blastdb_path_b = blastdb_path_2
 
      r = searchsorted(equiv_ends[root_b], pos_b)
      r = max(0, min(len(equiv_ends[root_b])-1, r))
      s2, e2, s1, e1 = equiv_regions[root_b][r]
      d2 = e2-pos_b
      pos_b = e1-d2
    
    seq_a = seq_dict_a[contig_a][start_a-1:end_a-1]
    seq_b = seq_dict_b[contig_b][start_b-1:end_b-1]
    
    delta = abs(pos_a-pos_b)
    
    if delta < close_cis:
      cc += 1
      print int(pair_id)*4, 'Ambig:', gen_a, gen_b  
      for label, seq, chromo, pos, blastdb, seq_dict in [('A:1', seq_a, chr_a, pos_a, blastdb_path_1, seq_dict_1),
                                                         ('A:2', seq_a, chr_a, pos_a, blastdb_path_2, seq_dict_2),
                                                         ('B:1', seq_b, chr_b, pos_b, blastdb_path_1, seq_dict_1),
                                                         ('B:2', seq_b, chr_b, pos_b, blastdb_path_2, seq_dict_2)]:
 
        result = BlastSearch(seq, blastdb, blastExe='blastn', eCutoff=1e-5, cpuCores=None).run().getBestHit()
        print '%s %s:%d' % (label, chromo, pos)
        if result:
          print '  %d-%d [%d]' % (result.queryFrom, result.queryTo, end_a-start_a)
          print seq
          print result.qseq
          print result.midline
          print result.hseq
          contig = result.name.split()[0]
 
          print contig, result.hitFrom, result.hitTo
          if result.hitFrom > result.hitTo:
            a = result.hitTo - result.queryFrom
            b = a + len(seq)
            print rev_comp(seq_dict[contig][a:b].upper())
 
          else:
            a = result.hitFrom - result.queryFrom
            b = a + len(seq)
            print seq_dict[contig][a:b]
 
        else:
          print "?"
          
      """
      for label, seq, chromo, pos in (('A', seq_a, chr_a, pos_a),
                                      ('B', seq_b, chr_b, pos_b)):
        
        result1 = BlastSearch(seq, blastdb_r1, blastExe='blastn',
                              eCutoff=1e-5, cpuCores=12).run()
        result2 = BlastSearch(seq, blastdb_r2, blastExe='blastn',
                              eCutoff=1e-5, cpuCores=12).run()
        
        hits1 = result1.hits if result1 else []
        hits2 = result2.hits if result2 else []
        seqs = []
        
        for i, hit in enumerate(hits1[:50]+hits2[:50]):
          contig = hit.name.split()[0]
          #print i+1, contig, hit.evalue
          #print hit.qseq
          #print hit.midline
          #print hit.hseq
          
          seqs.append((contig, hit.hseq))
     
        align_dict = multiAlign(seqs)
 
        seqs = sorted(align_dict.values())
        
        for s in seqs:
          print s
      """
          
    nm = 0
    
    for seq in (seq_a, seq_b):
      for letter in 'gcat': # repeat masked lowercase
        nm += seq.count(letter) 
    
    nm /= float(len(seq_a) + len(seq_b))
    
    x_vals.append(nm)
    y_vals.append(delta)

print cc
    
alpha = 0.2
ax1.scatter(x_vals, np.log10(y_vals), color='#0040FF', alpha=alpha)
ax1.set_xlabel('Rep. masked proportion')
ax1.set_ylabel('Diagonal sep.')

plt.show()

# Go through all SAM files.
#  - Collect polymorphisms from the mapping
#  - Are there any genomic sequences which are never seen perfectly?
#  - No evidence that they are still present

# For each read mapping/alignment
# - To both genomes separate chomosome entities
# - Record which base pairs are or are never perfectly matched

# Each genome can be done separately


