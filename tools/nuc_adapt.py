import os, glob, random, re
import numpy as np
import subprocess
import multiprocessing

from collections import defaultdict

PROG_NAME = 'nuc_adapt'
VERSION = '1.0.0'
DESCRIPTION = 'Adapt the sequence of a genome build using variants called from population Hi-C, or other related data'

DEFAULT_VCF = 'hic_comb_vars.vcf'
DEFAULT_QUAL = 30
PICARD    = '/home/tjs23/apps/picard.jar'
FREEBAYES = 'freebayes'
VCFUNIQ   = '/home/tjs23/apps/freebayes/vcflib/bin/vcfuniq'
 
def freebayes_genotype_job(region, genome_fasta_path, bam_paths):
  
  out_vcf_path = 'temp_%s_freebayes.vcf' % region
  
  if not os.path.exists(out_vcf_path):
 
    cmd_args = [FREEBAYES,
                '-f', genome_fasta_path,
                '-r', region,
                '-v', out_vcf_path] #, '--ploidy', '2']
 
    cmd_args += bam_paths
 
    util.call(cmd_args)

  return out_vcf_path
  

def call_genotype_freebayes(bam_file_paths, genome_fasta_path, out_vcf_path, min_qual, num_cpu):
  # FreeBayes pipeline
  
  temp_file_path = util.get_temp_path(out_vcf_path)

  # Make regions for parallelisation, splitting all chromos according to number of CPUs
  
  chromo_sizes = util.get_bam_chromo_sizes(bam_file_paths[0])
  
  regions = []
  region_fmt = '%s:%d-%d'
  
  for chromo, size in chromo_sizes:
    step = int(size/num_cpu) + 1 # will be rounded up
    
    i = 0
    j = step
    
    while j < size:
      regions.append(region_fmt % (chromo, i, j))
      i = j
      j += step
    
    regions.append(region_fmt % (chromo, i, size))
  
  # Call haplotype for all strains at once, split into parallel regions
  
  
  common_args = [genome_fasta_path, bam_file_paths]
  region_vcf_paths = util.parallel_split_job(freebayes_genotype_job, regions, common_args, num_cpu=num_cpu)
  
  # Combine the regions which were run in parallel
  
  util.info('Combining freebayes regions')
  
  with open(temp_file_path, 'w') as out_file_obj:
    write = out_file_obj.write
 
    for i, region_vcf in enumerate(region_vcf_paths):
      with open(region_vcf) as file_obj:
        for line in file_obj:
          if line[0] == '#':
            if i == 0:
              write(line)
 
          else:
            write(line)
  
  cmd_args = [VCFUNIQ]
  util.call(cmd_args, stdin=temp_file_path, stdout=out_vcf_path)
  
  # Cleanup temp files
  
  os.unlink(temp_file_path)

  for file_path in region_vcf_paths:
    os.unlink(file_path)
 
  return out_vcf_path


def _make_test_files(fasta_path='adapt_test.fasta', vcf_path='adapt_test.vcf',
                     min_len=200, max_len=1000, n_chromos=3, n_contigs=3):
  
  # Make a random FASTA file with 3 chromos and 3 contigs per chromo
    
  named_seqs = []

  var_dict = {}
  
  for i in range(n_chromos):
    chromo = 'chr_' + chr(ord('A') + i)
    
    for j in range(n_contigs):
      sl = random.randint(min_len, max_len)

      contig = '%s_contig%d' % (chromo, j)
      var_dict[contig] = []

      seq = ''.join(np.random.choice(['G','C','A','T'], sl).tolist())
      named_seqs.append((contig, seq))
      
      k = 0
      
      while k < sl-3:
        if seq[k:k+3] == 'GGG':
          var_dict[contig].append([k, 'GGG', 'g'])
          k += 3 
        
        elif random.random() < 0.025:
          var_dict[contig].append([k, seq[k], 'gcat'[random.randint(0,3)]])
          k += 2
        
        else:
          k += 1  
      
      var_dict[contig].sort() 
    
  util.write_fasta(fasta_path, named_seqs)
  
  with open(vcf_path, 'w') as file_obj:
    v = 0
    
    for contig in sorted(var_dict):
      for i, a, b in var_dict[contig]:
        data = [contig, '%d' % i, '%d' % v, a, b, '10', '-', '-\n']
        line = '\t'.join(data)
        file_obj.write(line)
        v += 1
        
  vcf_adapt_fasta(fasta_path, vcf_path, 'adapt_test_fixed.fasta')
  
  # Create pseudo VCF where all the GGG ->> TTT and CG -> GC etc
  # Check output with meld
  

def vcf_adapt_fasta(fasta_in_path, vcf_path, fasta_out_path, min_qual=DEFAULT_QUAL):
  
  msg = 'Reading VCF file %s' % vcf_path
  util.info(msg)
  
  vcf_dict = defaultdict(list)
  n_var = 0
  n = None
  
  with util.open_file(vcf_path) as file_obj:
    
    for line in file_obj:
      if line[0] == '#':
        continue
        
      else:
        contig, pos, v_id, ref, alt, qual, filt, info = line.split()[:8]
        
        if float(qual) < min_qual:
          continue
        
        pos = int(pos)
        vcf_dict[contig].append((pos, list(ref), list(alt)))
        n_var += 1

  msg = 'Sorting {:,} variations within {} segments'.format(n_var, len(vcf_dict))
  util.info(msg)
    
  vcf_dict = dict(vcf_dict)
  for chrom in vcf_dict:
    vcf_dict[chrom] = sorted(vcf_dict[chrom], reverse=True)

  msg = 'Adapting reference sequence in %s' % fasta_in_path
  util.info(msg)
  
  with util.open_file(fasta_in_path) as file_obj:
    inp_seqs = util.read_fasta(file_obj) # Chromo/contig_names must match
    inp_seqs.reverse() 
    out_seqs = []
    
    while inp_seqs:
      seq_id, seq = inp_seqs.pop()
      seq_vars = vcf_dict.get(seq_id, [])
      seq = list(seq)
      
      for i, ref_seq, alt_seq in seq_vars: # Last processed first
        n = len(ref_seq)
        seq[i:i+len(ref_seq)] = alt_seq
      
      seq = ''.join(seq)
      out_seqs.append((seq_id, seq))
      
    util.write_fasta(fasta_out_path, out_seqs)
      
      
def nuc_adapt(genome_fasta_path, hic_bam_paths, vcf_path, fasta_out_path=None,
              min_qual=DEFAULT_QUAL, num_cpu=None):
  # In future maybe do the clipping and genome mapping - get this code from NucProcess
  # Mapped reads should initially have been clipped at Hi-C ligation junction (separate ends)
  
  from nuc_tools import util, core, parallel
  
  if not num_cpu:
    num_cpu = parallel.MAX_CORES
  
  if not fasta_out_path:
    file_root, file_ext = os.path.splitext(genome_fasta_path)
    fasta_out_path = '%s_adapted%s' % (file_root, file_ext)  
  
  mdwm_bam_paths = []
  
  for bam_file_path in hic_bam_paths:
    util.info('Cleaning and deduplicating %s' % bam_file_path)
  
    file_root, file_ext = os.path.splitext(os.path.abspath(bam_file_path))
    
    clean_bam_path = '%s_temp%s' % (file_root, file_ext)  
    sort_bam_path = '%s_sort%s' % (file_root, file_ext)  
    mdwm_bam_path = '%s_mdwm%s' % (file_root, file_ext) 
    metrics_file_path= '%s_metrics.txt' % (file_root,) 
     
    cmd_args = ['samtools', 'view', '-b',
                '-@', str(num_cpu),
                '-f','3',
                '-F', '4',
                '-q','1', bam_file_path]
                
    # -f, 3 means include paired and mapped reads (first two bits of SAM format FLAG)
    # -F, 4 means exclude unmapped read (third bit of SAM format FLAG)
    # -q filters MAPQ: MAPping Quality. It equals -10log10 Pr{mapping position is wrong}, rounded to the nearest integer.
    
    if not os.path.exists(clean_bam_path):
      util.call(cmd_args, stdout=open(clean_bam_path, 'wb'))

    cmd_args = ['samtools', 'sort',
                '-O', 'bam',
                '-@', str(num_cpu),
                '-o', sort_bam_path, clean_bam_path]
    
    if not os.path.exists(sort_bam_path):
      util.call(cmd_args)
    
    
    """
    if  not os.path.exists(mdwm_bam_path):
      cwd = os.getcwd()
      os.chdir('/') # Picard picky about relative paths
 
      # Mark pair-end (position) read duplicates which are overwhelmingly repeat amplicons
      # best sequences are kept
      cmd_args = ['java', '-d64', '-Xmx4g', # 4 gigabyte heap size, for GATK, Picard etc.
                  '-jar', PICARD,
                  'MarkDuplicatesWithMateCigar',
                  'I=%s' % sort_bam_path,
                  'O=%s' % mdwm_bam_path,
                  'M=%s' % metrics_file_path]
 
      util.call(cmd_args)
      os.chdir(cwd)
      
    """
    
    #os.unlink(temp_bam_path)
    #os.unlink(sort_bam_path)
    #mdwm_bam_paths.append(mdwm_bam_path)    
    mdwm_bam_paths.append(sort_bam_path)    
  
  if not os.path.exists(vcf_path):
    call_genotype_freebayes(mdwm_bam_paths, genome_fasta_path, vcf_path, min_qual, num_cpu)
  
  vcf_adapt_fasta(genome_fasta_path, vcf_path, fasta_out_path, min_qual)
  
  util.info('Adapted genome sequence written to %s' % fasta_out_path)
  util.info('%s done!' % PROG_NAME)


# Dedup : 8,473,901 variations
# NDD: 8,473,901

def main(argv=None):
    
  from nuc_tools import util
  from argparse import ArgumentParser

  if argv is None:
    argv = sys.argv[1:]
  
  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'
  arg_parse = ArgumentParser(prog='nuc_tools adapt', description=DESCRIPTION,
                            epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument('bams', nargs='+', metavar='BAM_FILE',
                         help='One or more input BAM files obtained from mapping properly clipped Hi-C'
                              ' sequence reads to the reference genome build which is to be adapted')

  arg_parse.add_argument('-g', metavar='GENOME_FASTA_FILE',
                         help='Genome sequence as a single FASTA file, i.e. containing all chromosomes/contigs')

  arg_parse.add_argument('-q', metavar='MIN_QUALITY', type=int, default=DEFAULT_QUAL,
                         help='Minimum Phred scale quality score for selecting variations')

  arg_parse.add_argument('-v',  metavar='VCF_FILE', default=DEFAULT_VCF,
                         help='Optional output file path for VCF file containing combined variants called in input data. Default %s' % DEFAULT_VCF)

  arg_parse.add_argument('-o',  metavar='OUTPUT_FASTA_FILE',
                         help='Optional output file path for FASTA file containing adapted genome sequence. ' \
                              'Unless specified the output FASTA will be the input tagged with "_adapted"')  
 
  arg_parse.add_argument('-cpu', metavar='NUM_CORES', default=util.parallel.MAX_CORES, type=int,
                         help='Number of parallel CPU cores to use for variant calling. Default: All available (%d)' % util.parallel.MAX_CORES) 

  args = vars(arg_parse.parse_args(argv))
  
  hic_bam_paths     = args['bams']
  genome_fasta_path = args['g']
  min_qual          = args['q']
  vcf_out_path      = args['v']
  fasta_out_path    = args['o']
  num_cpu           = args['cpu'] or None # May not be zero
  
  nuc_adapt(genome_fasta_path, hic_bam_paths, vcf_out_path, fasta_out_path, min_qual, num_cpu)
    
if __name__ == '__main__':

  
  #_make_test_files()
  #import sys
  #sys.exit()
  
  main()
