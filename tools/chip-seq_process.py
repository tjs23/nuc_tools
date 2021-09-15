import sys, os
from collections import defaultdict
from glob import glob

PROG_NAME = 'chip_seq'
VERSION = '1.0.0'
DESCRIPTION = 'ChIP-seq read processing and peak calling pipeline'

ADAPTER_SEQS = {'Nextera':'CTGTCTCTTATA',
                'Illumina universal':'AGATCGGAAGAGC'}

DEFAULT_ADAPTER = 'Illumina universal'

QUAL_SCHEMES = ['phred33', 'phred64', 'solexa']

BOWTIE2_QUAL_SCHEMES = {'phred33':'--phred33',
                        'phred64':'--phred64',
                        'solexa':'--solexa-quals'}

MIN_READ_LEN = 20

DEFAULT_MIN_QUAL = 10

QUAL_ZERO_ORDS = {'phred33':33, 'phred64':64, 'solexa':64}

FILE_BUFFER = 2**16

MIN_ADAPT_OVERLAP = 7

DEFAULT_MAX_SEP = 500

def _check_index_file(file_path, sub_files=('.1', '.2', '.3', '.4', '.rev.1', '.rev.2')):

  from nuc_tools import io, util
  
  msg = ''
  is_ok = True

  if os.path.exists(file_path + '.1.bt2l'):
    file_ext = '.bt2l' # All build files should be long
  else:
    file_ext = '.bt2'

  for sub_file in sub_files:
    full_path = file_path + sub_file + file_ext
    is_ok, msg = io.check_regular_file(full_path)
    
    if not is_ok:
      msg = 'Genome index error. ' + msg    
      break

  if not is_ok:
    util.critical(msg)


def clip_reads(in_fastq_file, out_fastq_file, qual_scheme, min_qual=DEFAULT_MIN_QUAL,
               adapt_seqs=None, min_len=MIN_READ_LEN):
  """
  Clips reads at poor quality base calls and adapter sequences.
  Discards very short reads. 
  """
  from nuc_tools import util, io
  
  util.info('Clipping FASTQ file %s, producing %s' % (in_fastq_file, out_fastq_file))
  
  adapt_seqs = adapt_seqs or []
  
  n_reads = 0
  n_qclip = 0
  n_short = 0
  n_adapt = 0
  mean_len = 0
  
  zero_ord = QUAL_ZERO_ORDS[qual_scheme]
  
  # gzip agnostic input
  with io.open_file(in_fastq_file) as in_file_obj, open(out_fastq_file, 'w', FILE_BUFFER) as out_file_obj:
    readline = in_file_obj.readline
    write = out_file_obj.write
    
    line1 = readline()
    while line1[0] != '@':
      line1 = readline()

    while line1:
      n_reads += 1
      line2 = readline()[:-1]
      line3 = readline()
      line4 = readline()[:-1]
      
      # Quality clip
      q = 0
      while line2 and line2[-1] == 'N':
        q = 1
        line2 = line2[:-1]
        line4 = line4[:-1]

      while line4 and (ord(line4[-1]) - zero_ord) < min_qual:
        q = 1
        line2 = line2[:-1]
        line4 = line4[:-1]

      while line4 and (ord(line4[0]) - zero_ord) < min_qual:
        q = 1
        line2 = line2[1:]
        line4 = line4[1:]

      n_qclip += q
      
      # Adapter clip
      for adapt_seq in adapt_seqs:
        if adapt_seq[:MIN_ADAPT_OVERLAP] in line2:
          i = line2.index(adapt_seq[:MIN_ADAPT_OVERLAP])
 
          if line2[i:i+len(adapt_seq)] in adapt_seq:
            line2 = line2[:i]
            line4 = line4[:i]
            n_adapt += 1
            break
      
      n = len(line2)
      
      if n < min_len: # Check size
        n_short += 1

      mean_len += n
      write('%s%s\n%s%s\n' % (line1, line2, line3, line4))
 
      #if max_reads_in and n_reads >= max_reads_in:
      #  break
 
      line1 = readline()
  
  mean_len /= float(n_reads-n_short)

  util.info(' .. num input reads: %d' % n_reads,)
  util.info(' .. quality clipped: %d' % n_qclip)
  util.info(' .. adapter clipped: %d' % n_adapt)
  util.info(' .. too short: %d' % n_short)
  util.info(' .. mean length: %.2f' % mean_len)
  
  return n_reads, n_qclip, n_adapt, n_short, mean_len


def read_chromo_names(file_path):

  from nuc_tools import util, io  
 
  name_dict = {}
      
  with io.open_file(file_path) as file_obj:
    for line in file_obj:
      line = line.strip()
 
      if not line:
        continue
 
      if line[0] == '#':
        continue
 
      data = line.split()
 
      if len(data) < 2:
        continue
 
      else:
        contig, name = data[:2]
 
        if contig in name_dict:
          msg ='Chromosome naming file "%s" contains a repeated sequence/contig name: %s'
          util.critical(msg % (file_path, contig))

        name_dict[contig] = name
 
  if not name_dict:
    util.critical('Chromosome naming file "%s" contained no usable data: requires whitespace-separated pairs' % file_path)      
  
  return name_dict
  
  
def chip_seq_process(fastq_path_groups, sample_names, genome_index, out_dir=None, control_fastq_paths=None,
                     control_name=None, control_bam_path=None, chromo_names_path=None, align_exe=None,
                     qual_scheme=None, min_qual=DEFAULT_MIN_QUAL, max_sep=DEFAULT_MAX_SEP,
                     adapt_seqs=None, num_cpu=None, keep_macs=False, full_out=False, frag_size=None):

  from nuc_tools import util, io, formats, parallel
  import shutil
  
  if not (0 <= min_qual <= 40):
    util.critical('Miniumum FASTQ quality score must be in the range 0-40 (%d specified).' % min_qual)
  
  if out_dir:
    if not os.path.exists(out_dir):
      util.critical('Output directory %s does not exist' % out_dir)
  
  else:
    out_dir = '' # CWD default    
  
  if control_fastq_paths and control_bam_path:
    util.critical('cannot set both control_fastq_paths and control_bam_path')
    
  if not control_fastq_paths and not control_bam_path:
    util.critical('must set one of control_fastq_paths and control_bam_path')

  if not control_name:
    path = control_fastq_paths[0] if control_fastq_paths else control_bam_path
    file_root =  os.path.splitext(os.path.basename(path))[0]
    control_name = 'C%s' % file_root

  sample_names = list(sample_names) # can be a tuple but need to modify below

  # How is the library size (c.f. discordants) accounted for...
  
  # Check FASTQ files and corresponding sample names
  treatment_fastq_1 = None
  from formats import fastq
  for i, fastq_paths in enumerate(fastq_path_groups):
    for file_path in fastq_paths:
      #io.check_regular_file(file_path, critical=True)
      io.check_regular_file(file_path)
      formats.fastq.check_format(file_path)  
      if not treatment_fastq_1:
        treatment_fastq_1 = file_path
    
    if not sample_names[i]:
      file_root =  os.path.splitext(os.path.basename(fastq_paths[0]))[0]
      sample_names[i] = 'S%d_%s' % (i+1, file_root)
    
  # Check genome index
  _check_index_file(genome_index)
  
  # Check chromo naming file
  
  if chromo_names_path:
    io.check_regular_file(chromo_names_path, critical=True)   
    chromo_name_dict = read_chromo_names(chromo_names_path)
  else:
    chromo_name_dict = {}
      
  # Check aligner
  if not align_exe:
    align_exe = io.locate_exe('bowtie2')

  if not align_exe:
    msg = 'Aligner bowtie2 could not be found'
    util.critical(msg)

  elif not os.path.exists(align_exe):
    msg = 'Aligner executable path "%s" not found'
    util.critical(msg % align_exe)

  else:
    #io.check_regular_file(align_exe, critical=True)      
    io.check_regular_file(align_exe)
  
  # Check samtools
  samtools_exe = io.locate_exe('samtools')
  if not samtools_exe:
    msg = 'Samtools executable could not be found'
    util.critical(msg)  
  
  # Check macs2
  macs2_exe  = io.locate_exe('macs2')
  if not macs2_exe:
    msg = 'Macs2 executable could not be found'
    util.critical(msg)  
   
  # Check FASTQ quality scheme
  if qual_scheme:
    if qual_scheme not in QUAL_SCHEMES:
      msg = 'FASTQ quality scheme "%s" not known. Available: %s.' % (qual_scheme, ', '.join(sorted(QUAL_SCHEMES)))
      msg += ' Scheme will be deduced automatically if not specified.'
      util.critical(msg)

  else:
    qual_scheme = formats.fastq.get_qual_scheme(fastq_paths[0])  
  
  if control_bam_path:
    io.check_regular_file(control_bam_path)
    
  if not num_cpu:
    num_cpu = parallel.MAX_CORES
  
  map_args = [align_exe, '-D', '20', '-R', '3', '-N', '0',  '-L', '20',  '-i', 'S,1,0.5', # similar to very-sensitive
              '-x', genome_index, # '-k', '2',
              '--reorder', # '--score-min', 'L,-0.6,-0.6',
              '-p', str(num_cpu),
              '-X', str(max_sep)]  
  
  if control_fastq_paths:
    path_root = os.path.join(out_dir, control_name)
    
    if not control_bam_path:
      control_bam_path = path_root + '.bam'
    
    control_sam_path = path_root + '.sam'
   
    if len(control_fastq_paths) == 2:
      fastq_1, fastq_2 = control_fastq_paths
      
      control_fastq_1 = io.tag_file_name(path_root, 'clip_1', '.fastq')
      control_fastq_2 = io.tag_file_name(path_root, 'clip_2', '.fastq')      
      
      if not os.path.exists(control_fastq_1):
        clip_reads(fastq_1, control_fastq_1, qual_scheme, min_qual, adapt_seqs)

      if not os.path.exists(control_fastq_2):
        clip_reads(fastq_2, control_fastq_2, qual_scheme, min_qual, adapt_seqs)
    
      util.info('Mapping control paired-end FASTQ reads to genome index %s' % genome_index)
      
      cmd_args = map_args + ['-1', control_fastq_1, '-2', control_fastq_2, '-S', control_sam_path]
      f_flag = '3' 
    else:
      fastq_1 = control_fastq_paths[0]
      
      control_fastq_1 = io.tag_file_name(path_root, 'clip_1', '.fastq')
      
      if not os.path.exists(control_fastq_1):
        clip_reads(fastq_1, control_fastq_1, qual_scheme, min_qual, adapt_seqs)
      
      util.info('Mapping control single-end FASTQ reads to genome index %s' % genome_index)
      
      cmd_args = map_args + ['-U', control_fastq_1, '-S', control_sam_path]
      #f_flag = '2'
      f_flag = None
      
    cmd_args.append(BOWTIE2_QUAL_SCHEMES[qual_scheme])
    util.call(cmd_args)

    util.info("Converting SAM file output into sorted BAM")
  
    cmd_args = [samtools_exe, 'sort', '-O', 'bam', # '-@', str(num_cpu), # option only avail in newer samtools
                '-o', control_bam_path,  control_sam_path] 
    util.call(cmd_args)
    
    os.unlink(control_sam_path)    
                 
  elif control_bam_path:
    #io.check_regular_file(control_bam_path, critical=True)     
    io.check_regular_file(control_bam_path)
    #fastq_1 = control_bam_path[:-4] + '.fq'
  
  
  g = 0
  for sample_name, fastq_paths in zip(sample_names, fastq_path_groups):
    path_root = os.path.join(out_dir, sample_name)
    
    g += 1
    nfq = len(fastq_paths)
    
    if not 0 < nfq < 3:
      util.critical('One or two FASTQ files must be specified for group %d. Found: %d' % (g, nfq))
    
    clean_bam_file_path = io.tag_file_name(path_root, 'clean', '.bam')
    
    if not os.path.exists(clean_bam_file_path):
      sam_file_path_temp  = io.tag_file_name(path_root, util.TEMP_ID, '.sam')
      bam_file_path_temp  = io.tag_file_name(path_root, util.TEMP_ID, '.bam')
      
      # extract gzipped and clip FASTQ
      clip_fastq_paths = []
      for i, in_fq in enumerate(fastq_paths):
        clip_fastq_path = io.tag_file_name(path_root, 'clip_%d' % (i+1), '.fastq')
        clip_fastq_paths.append(clip_fastq_path)
        clip_reads(in_fq, clip_fastq_path, qual_scheme, min_qual, adapt_seqs)

 
      if nfq == 2:
        util.info('Mapping ChIP paired-end FASTQ reads to genome index %s' % genome_index)
        cmd_args = map_args + ['-1', clip_fastq_paths[0], '-2', clip_fastq_paths[1], '-S', sam_file_path_temp]
        f_flag = '3' # Extra paired check in samtools filtering
 
      else:
        util.info('Mapping ChIP single-end FASTQ reads to genome index %s' % genome_index)
        cmd_args = map_args + ['-U', clip_fastq_paths[0], '-S', sam_file_path_temp]
        #f_flag = '2'
        f_flag = None
 
      cmd_args.append(BOWTIE2_QUAL_SCHEMES[qual_scheme])
      util.call(cmd_args)
 
      util.info("Converting SAM file output into sorted BAM")
 
      cmd_args = [samtools_exe, 'sort', # '-O', 'bam', # option only avail in newer samtools
                  '-o', bam_file_path_temp,  sam_file_path_temp]
      util.call(cmd_args)
 
      os.unlink(sam_file_path_temp)
 
      util.info('Removing unmapped and low quality read alignments')
 
      # -f : must have these bits ; 2 = properly aligned, accounting for any pairs, 1 = part of a read pair
      # -F : must not have these bits ; 4 = unmapped
      # -q : quality
 
      #cmd_args = [samtools_exe,'view','-b','-f',f_flag,'-F','4','-q','30', bam_file_path_temp]
      cmd_args = [samtools_exe,'view','-b', '-F','4','-q','30']
      if f_flag:
        cmd_args.extend(['-f', f_flag])
      cmd_args.append(bam_file_path_temp)
 
      with open(clean_bam_file_path, 'wb') as file_obj:
        util.call(cmd_args, stdout=file_obj)
 
      os.unlink(bam_file_path_temp)
 
      util.info('Indexing BAM file')
 
      cmd_args = [samtools_exe, 'index', clean_bam_file_path]
      util.call(cmd_args)
    
    from formats import sam
    chromo_sizes = formats.sam.get_bam_chromo_sizes(clean_bam_file_path)
    
    contigs, sizes = zip(*chromo_sizes)
    
    genome_size = '%.2e' % sum(sizes)
    
    # MACS2
    
    
    peak_out_dir = os.path.join(os.path.dirname(treatment_fastq_1), 'macs2_peaks_%s_%s' % (sample_name, util.get_rand_string(5)))
    io.makedirs(peak_out_dir, exist_ok=True)
    
    broad_name = sample_name + '_b'
    narrow_name = sample_name + '_n'
    
    broad_bed_in = os.path.join(peak_out_dir,'%s_peaks.broadPeak' % broad_name)
    narrow_bed_in = os.path.join(peak_out_dir,'%s_peaks.narrowPeak' % narrow_name)
    
    broad_bed_out = io.tag_file_name(path_root, 'broad', '.bed')
    narrow_bed_out = io.tag_file_name(path_root, 'narrow', '.bed')
    
    if not os.path.exists(narrow_bed_out):
      if nfq == 2:
        fmt = 'BAMPE'
        
        if frag_size:
          util.warn('DNA fragment size specification (-fs) ignored for paired end reads' )
          frag_size = None
        
      else:
        fmt = 'BAM'
 
      buffer_size = max(int(1e4), max(sizes)/1000)
 
      common_args = [macs2_exe, 'callpeak',
                     '-t', clean_bam_file_path,
                     '--buffer-size', str(buffer_size), # Need to reduce when large number of contigs/chromos
                     '-f', fmt,
                     '-g', genome_size]
 
      if control_bam_path:
        common_args += ['-c', control_bam_path]
      
      if frag_size:
        common_args += ['--nomodel', '--extsize', str(frag_size)]
      
      cmd_args = common_args + ['-n', broad_name, '-B', '-q', '0.05', '--broad', '--outdir', peak_out_dir]
      util.info('Calling broad peaks')
      util.call(cmd_args)
 
      cmd_args =  common_args + ['-n', narrow_name, '-B', '-q', '0.05', '--outdir', peak_out_dir]
      util.info('Calling narrow peaks')
      util.call(cmd_args)

      # Collate and rename outputs
 
      #broad_bed_out = io.tag_file_name(fastq_1, 'broad', '.bed')
      #narrow_bed_out = io.tag_file_name(fastq_1, 'narrow', '.bed')
 
      for bed_in, bed_out in ((broad_bed_in, broad_bed_out),
                              (narrow_bed_in, narrow_bed_out)):
 
        with io.open_file(bed_in) as in_file_obj, open(bed_out, 'w', FILE_BUFFER) as out_file_obj:
          write = out_file_obj.write
          join = '\t'.join
 
          for line in in_file_obj:
            data = line.split()
            contig = data[0]
            chromo = chromo_name_dict.get(contig, contig)
 
            if len(chromo) < 3:
              chromo = 'chr' + chromo
 
            data[0] = chromo
            if not full_out:
              data = data[:5]
 
            write(join(data) + '\n')
 
        util.info('Written BED file %s' % bed_out)
 
      if not keep_macs:
        util.info('Cleanup MACS2 files, removing %s' % peak_out_dir)
        shutil.rmtree(peak_out_dir)
  
  # Ad sample/file_tag names
    
      
def main(argv=None):
  
  from argparse import ArgumentParser
  from nuc_tools import util, io

  if argv is None:
    argv = sys.argv[1:]
  
  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'
  arg_parse = ArgumentParser(prog='nuc_tools ' + PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
                             
  arg_parse.add_argument('-f1', nargs='+', metavar='FASTQ_FILE',
                         help='First (mandatory) ChIP FASTQ sequence read files to process.' \
                              ' May be single end or paired end. Accepts wildcards' \
                              ' that match one or two files. Paired end data is assumed if two file paths are specified.' \
                              ' Optional sample name (e.g. indicating the ChIPed protein) to label output files may be' \
                              ' specified using the name@file_path format, for example "SAMPLE_A@seq_data_reads.fq" will generate SAMPLE_A_broad.bed, SAMPLE_A_narrow.bed etc.')

  arg_parse.add_argument('-f2', nargs='+', metavar='FASTQ_FILE',
                         help='Second (optional) ChIP FASTQ sequence read files to process.' \
                              ' May be single end or paired end. Accepts wildcards' \
                              ' that match one or two files. Paired end data is assumed if two file paths are specified.'\
                              ' Optional sample name to label output files may be specified using the name@file_path format.' \
                              ' NOTE: an arbitrary number of input FASTQ groups may be specified with further numbered options -f3, -f4 etc..')
   
  arg_parse.add_argument('-c', '--control-fastq', nargs='+', metavar='FASTQ_FILE', dest='c',
                         help='Control FASTQ sequence read files to process (i.e sample input without antibody).' \
                              ' Optional name to label output files may be specified using the name@file_path format.' \
                              ' May be single end or paired end. Accepts wildcards' \
                              ' that match one or two files. Paired end data is assumed if two file paths are specified.' \
                              ' Pre-mapped reads may be specified instead using the -cb option.')

  arg_parse.add_argument('-cb', '--control-bam', default=None, metavar='BAM_FILE', dest='cb',
                         help='Optional file path for the control sample BAM format alignments' \
                              ' (i.e sample input without antibody) to compare with ChIP reads. ' \
                              ' Optional name to label output files may be specified using the name@file_path format.' \
                              ' If control FASTQ files are specified (-c) this file will be created (and potentially overwritten).' \
                              ' Otherwise this file should exist and contain pre-mapped reads in BAM format.')

  arg_parse.add_argument('-o', '--out-dir', metavar='DIR_PATH', dest='o',
                         help='Output directory for writing output files. Optional and defaults to the current working directory.')

  arg_parse.add_argument('-m', '--max-pair-sep', metavar='MAX_SEPARATION', dest='m', default=DEFAULT_MAX_SEP,
                         type=int, help='Maximum base pair separation between paired reads, i.e. maximum fragment size in DNA library/sample. Default: %d' % DEFAULT_MAX_SEP)
  
  arg_parse.add_argument('-g', '--gnome-index', metavar='GENOME_FILE', dest='g',
                         help='Location of genome index files to map sequence reads to, without' \
                              ' any file extensions like ".1.b2" etc.')

  arg_parse.add_argument('-q', '--qual-scheme', metavar='SCHEME', dest='q',
                         help='Use a specific FASTQ quality scheme (normally not set' \
                              ' and deduced automatically). Available: %s' % ', '.join(QUAL_SCHEMES))

  arg_parse.add_argument('-cn', '--chromo-names', default=None, metavar='CHROM_NAME_FILE', dest='cn',
                         help='Location of a file containing chromosome names for the genome build:' \
                              ' tab-separated lines mapping sequence/contig names (as appear at the' \
                              ' start of genome FASTA headers) to desired (human readable) chromosome' \
                              ' names. The file may be built automatically from NCBI' \
                              ' genome FASTA files using the "nuc_sequence_names" program')  

  arg_parse.add_argument('-fo', '--full-output', default=False, action='store_true', dest='fo',
                         help='Output BED format files with full MACS2 score columns.')

  arg_parse.add_argument('-k', '--keep-macs', default=False, action='store_true', dest='k',
                         help='Keep all MACS2 peak-calling output files.')

  arg_parse.add_argument('-qm', '--qual-min', default=DEFAULT_MIN_QUAL, metavar='MIN_QUALITY', type=int, dest='qm',
                         help='Minimum acceptable FASTQ quality score in range 0-40 for' \
                              ' clipping ends of reads. Default: %d' % DEFAULT_MIN_QUAL)
                              
  arg_parse.add_argument('-fs', '--frag-size', default=0, metavar='BP_LENGTH', dest='fs',
                         type=int, help='Fix the MACS2 molecule fragment size (for single-end reads only), and do not estimate from the reads.')
                        
  arg_parse.add_argument('-b', '--bowtie2-path', metavar='EXE_FILE', dest='b',
                         help='Path to bowtie2 (read aligner) executable: will be searched' \
                              ' for if not specified.')

  arg_parse.add_argument('-n', '--num-cpu', default=0, metavar='CPU_COUNT', dest='n',
                         type=int, help='Number of CPU cores to use in parallel for genome mapping. Defaults to all available.')

  default_ad_seq = ADAPTER_SEQS[DEFAULT_ADAPTER]
  ad_prests = ', '.join(['%s:%s' % (k, v) for k, v in ADAPTER_SEQS.items()])
  
  arg_parse.add_argument('-ad', '--adapter-seq', nargs='*',  default=[default_ad_seq],
                         metavar='ADAPTER_SEQ', dest='ad',
                         help='Adapter sequences to truncate reads at (or blank for none). E.g. %s. ' \
                         'Default: %s (%s)' % (ad_prests, ADAPTER_SEQS[DEFAULT_ADAPTER], DEFAULT_ADAPTER))                         
  
  parsed, unknown = arg_parse.parse_known_args()
  
  for arg in unknown:
    if arg.startswith('-f') and arg[2:].isdigit():
      arg_parse.add_argument(arg, nargs='+', metavar='FASTQ_FILES')
  
  args = vars(arg_parse.parse_args(argv))
  fastq_inputs = []

  for arg in args:
    if arg.startswith('f') and arg[1:].isdigit() and args[arg]:
      file_paths = []
      file_path = args[arg][0]
      
      if '@' in file_path:
        k = file_path.rfind('@')
        sample_name = file_path[:k]
        
        for file_path in args[arg]:
         file_paths += glob(file_path[k+1:])
         
      else:
        sample_name = None
      
      fastq_inputs.append((file_paths, sample_name))
     
  if not fastq_inputs:
    util.critical('No ChIP FASTQ files specified')
  
  fastq_path_groups, sample_names = zip(*fastq_inputs)
     
  genome_index = args['g']
  out_dir = args['o']
  control_fastqs = args['c']
  control_bam = args['cb']
  chromo_names = args['cn']
  qual_scheme = args['q']
  min_qual = args['cn']
  full_out = args['fo']
  keep_macs = args['k']
  min_qual = args['qm']
  max_sep = args['m']
  frag_size = args['fs']
  align_exe = args['b']
  num_cpu = args['n']
  adapt_seqs = args['ad'] or []
   
  if control_bam:
    if '@' in control_bam:
      k = control_bam.rfind('@')
      control_name = control_bam[:k]
      control_bam = control_bam[k+1:]
    else:
      control_name = None
      
  elif control_fastqs:
    file_paths = []
    file_path = control_fastqs[0]
    
    if '@' in file_path:
      k = file_path.rfind('@')
      control_name = file_path[:k]
      
      for file_path in control_fastqs:
        file_paths += glob(file_path[k+1:])
      
      control_fastqs = file_paths
    else:
      control_name = None
    
  else:
    util.critical('No ChIP control/input files specified (in BAM or FASTQ format)')
   
  chip_seq_process(fastq_path_groups, sample_names,
                   genome_index, out_dir, control_fastqs,
                   control_name, control_bam, chromo_names, align_exe,
                   qual_scheme, min_qual, max_sep, adapt_seqs, num_cpu,
                   keep_macs, full_out, frag_size)
   
if __name__ == '__main__':

  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
  main()


