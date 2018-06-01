import subprocess


# #   Globals  # #


# #   BAM/SAM format  # #

def get_bam_chromo_sizes(bam_file_path):

  # Looks in header of BAM file to get chromosome/contig names and their lengths  
    
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
