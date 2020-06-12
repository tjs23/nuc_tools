import subprocess, io, os
import numpy as np

from nuc_tools import util

from collections import defaultdict

def load_data_track(file_path, bin_size=1000, min_qual=10):
  
  chromos_sizes = dict(get_bam_chromo_sizes(file_path))

  data_hists_pos = {c: np.zeros(int(chromos_sizes[c]//bin_size)+1, 'uint16') for c in chromos_sizes}
  data_hists_neg = {c: np.zeros(int(chromos_sizes[c]//bin_size)+1, 'uint16') for c in chromos_sizes}
 
  cmd_args = ['samtools', 'view','-F','4','-q', str(min_qual), file_path]
  
  proc = subprocess.Popen(cmd_args, shell=False,
 			  stdout=subprocess.PIPE)
 
  util.info('Reading {}'.format(file_path))
 
  #for line in io.TextIOWrapper(proc.stdout, encoding='ascii'):
  n = 0
  for line in proc.stdout:
 
    rname, sam_flag, chromo, pos, mapq, cigar, mate_contig, mate_pos, t_len, seq, qual = line.split('\t')[:11]
    idx = int(int(pos)//bin_size)
 
    if int(sam_flag) & 0x10:
      data_hists_neg[chromo][idx] += 1
 
    else:
      data_hists_pos[chromo][idx] += 1
    
    n += 1
    
  data_dict = defaultdict(set)
  label = ''
 
  for strand, data_hists in (('+', data_hists_pos), ('-', data_hists_neg)):
    for chromo in data_hists:
      hist = data_hists[chromo]
      idx = hist.nonzero()[0]
      seq_len = chromos_sizes[chromo]
      add = data_dict[chromo].add
 
      for i in idx:
 	start = i * bin_size
 	end = start+bin_size-1
 	value = hist[i]
 	add((start, end, strand, value, value, label))

  util.info(' .. processed {:,} BAM/SAM entries'.format(n))

  return util.finalise_data_track(data_dict)
 
  
def get_bam_chromo_sizes(bam_file_path):

  # Looks in header of BAM file to get chromosome/contig names and their lengths  
  
  if not os.path.exists(bam_file_path + '.bai'):
    sort_path = os.path.splitext(bam_file_path)[0] + '_sort.bam'
    
    if not os.path.exists(sort_path + '.bai'):
      util.info('Sorting and indexing {}'.format(bam_file_path))
      cmd_args = ['samtools', 'sort', bam_file_path, '-o', sort_path]
      proc = subprocess.Popen(cmd_args, shell=False,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
 
      std_out_data, std_err_data = proc.communicate()

      cmd_args = ['samtools', 'index', sort_path]
      proc = subprocess.Popen(cmd_args, shell=False,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
 
      std_out_data, std_err_data = proc.communicate()
    
    bam_file_path = sort_path

  cmd_args = ['samtools', 'idxstats', bam_file_path]
  
  proc = subprocess.Popen(cmd_args, shell=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
                          
  std_out_data, std_err_data = proc.communicate()
  
  
  chromos_sizes = [] 
  
  for line in std_out_data.decode('ascii').split('\n'):
    if line:
      ref_name, seq_len, n_mapped, n_unmapped = line.split()
      seq_len = int(seq_len)
      
      if seq_len:
        chromos_sizes.append((ref_name, seq_len))
  
  
  return chromos_sizes
