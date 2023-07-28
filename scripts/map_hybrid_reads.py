# Use Bowtie 2 mapping c.f. NucProcess

import os
import re
import string
import sys
import gzip

from io import BufferedReader
from math import floor
from shutil import move, which
from subprocess import Popen, PIPE, call

import numpy as np

VERSION = '0.1.1'
SESSION_KEY = ''.join(np.random.choice(tuple(string.ascii_letters), 8))
TEMP_EXT = '_temp_%s' % (SESSION_KEY)
READ_BUFFER = 2**16
QUAL_ZERO_ORDS = {'phred33':33, 'phred64':64, 'solexa':64}
BOWTIE_MAX_AMBIG_SCORE_TOL = 5
ID_LEN = 10
SCORE_TAG = re.compile(r'\sAS:i:(\S+).+\sMD:Z:(\S+)')
SCORE_TAG_SEARCH = SCORE_TAG.search
CLOSE_AMBIG = 1000
MIN_READ_LEN = 18
MIN_ADAPT_OVERLAP = 7
ADAPTER_SEQS = {'Nextera':'CTGTCTCTTATA',
                'Illumina universal':'AGATCGGAAGAGC',
                'Tn5_ME': 'CTGTCTCTTATACACATCT',
                #'Tn5_MEr':'AGATGTGTATAAGAGACAG',
                }
                                  
                                  
def info(msg, prefix='INFO'):

  print('%s: %s' % (prefix, msg))


def warn(msg, prefix='WARNING'):

  print('%s: %s' % (prefix, msg))


def fatal(msg, prefix='FAILURE'):
  
  print('%s: %s' % (prefix, msg))
  sys.exit(0)


def tag_file_name(file_path, tag, file_ext=None, sep='_', ncc_tag='_nuc'):

  if tag:
    tag = sep + tag

  dir_name, file_name = os.path.split(file_path)

  if file_name.endswith('.gz'):
    file_root, file_ext_old = os.path.splitext(file_name[:-3])
    file_name = file_root + tag + (file_ext or file_ext_old) + '.gz'

  elif file_name.endswith('.ncc') or (file_ext == '.ncc'):
    file_root, file_ext_old = os.path.splitext(file_name)

    if '_ambig' in file_root:
      ambig_tag = '_ambig'
    else:
      ambig_tag = ''

    if ncc_tag in file_root:
      file_root = file_root[:file_root.rindex(ncc_tag)]

    file_name = file_root + ncc_tag + tag + ambig_tag + (file_ext or file_ext_old)

  else:
    file_root, file_ext_old = os.path.splitext(file_name)
    file_name = file_root + tag + (file_ext or file_ext_old)

  file_path = os.path.join(dir_name, file_name)

  return file_path

def open_file_r(file_path, complete=True, gzip_exts=('.gz','.gzip'), buffer_size=READ_BUFFER):

  import io, subprocess
 
  if os.path.splitext(file_path)[1].lower() in gzip_exts:

    if complete:
      try:
        file_obj = subprocess.Popen(['zcat', file_path], stdout=subprocess.PIPE).stdout
      except OSError:
        file_obj = BufferedReader(gzip.open(file_path, 'rb'), buffer_size)
   
    else:
      file_obj = BufferedReader(gzip.open(file_path, 'rb'), buffer_size)
   
    file_obj = io.TextIOWrapper(file_obj, encoding="utf-8")

  else:
    file_obj = open(file_path, 'r', buffer_size, encoding='utf-8')
 
  return file_obj

def _skip_sam_header(readline):

  line = readline()

  while line and line[0] == '@':
    line = readline()

  return line
  
  
def clip_reads(fastq_file, file_root, qual_scheme, min_qual,
               max_reads_in=None, adapt_seqs=None, trim_5=0, trim_3=0,
               is_second=False, min_len=MIN_READ_LEN):
  """
  Clips reads at ligation junctions, removes anbigous ends and discards very short reads
  """
  
  job = 2 if is_second else 1
  tag = 'reads2_clipped' if is_second else 'reads1_clipped'
  adapt_seqs = adapt_seqs or list(ADAPTER_SEQS.values())
        
  sam_file_path = tag_file_name(file_root, '%s_map%d' % (tag, job), '.sam')
  sam_file_path_temp = sam_file_path + TEMP_EXT
  
  clipped_file = tag_file_name(file_root, tag, '.fastq')
  
  if os.path.exists(clipped_file):
    print(f'Found existsing {clipped_file}')
    return clipped_file
  
  clipped_file_temp = clipped_file + TEMP_EXT
  
  in_file_obj = open_file_r(fastq_file, complete=not bool(max_reads_in))
  
  #junct_seq = 'CTGTCTCTTATACACATCT'
  #junc_len =len(junct_seq)
  #n_jclip = 0
  
  n_reads = 0
  n_qclip = 0
  n_short = 0
  n_adapt = 0
  mean_len = 0
  zero_ord = QUAL_ZERO_ORDS[qual_scheme]

  out_file_obj = open(clipped_file_temp, 'w', READ_BUFFER)
  write = out_file_obj.write
  readline = in_file_obj.readline

  line1 = readline()
  while line1[0] != '@':
    line1 = readline()
  
  end = -1 - trim_3
  
  adapt_list = [(adapt_seq, adapt_seq[:MIN_ADAPT_OVERLAP], len(adapt_seq)) for adapt_seq in adapt_seqs]
  
  while line1:
    n_reads += 1
    line2 = readline()[trim_5:end]
    line3 = readline()
    line4 = readline()[trim_5:end]   
   
    #if junct_seq in line2:
    #  n_jclip += 1
    #  i = line2.index(junct_seq)
    #  line2 = line2[:i] + line2[i+junc_len:]
    #  line4 = line4[:i] + line4[i+junc_len:]
    
    for adapt_seq, min_adapt, alen in adapt_list:
    
      if min_adapt in line2:
        i = line2.index(min_adapt)
        adapt_end = line2[i:i+alen]
        
        if adapt_end in adapt_seq:
          line2 = line2[:i]
          line4 = line4[:i]
          n_adapt += 1
        
        else:
          n_mismatch = 0
          for j, bp in enumerate(adapt_end):
            if bp == 'N':
              continue
            
            elif bp != adapt_seq[j]:
              n_mismatch += 1
          
          if n_mismatch < 2:
            line2 = line2[:i]
            line4 = line4[:i]
            n_adapt += 1          
           
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
    
    for i, qs in enumerate(line4):
      if (ord(qs) - zero_ord) < min_qual:
        line2 = line2[:i] + 'N' + line2[i+1:]
    
    n_qclip += q

    n = len(line2)
    
    if n < min_len:
      n_short += 1

    else:
      mean_len += len(line2)
      line1 = '@%10.10d_%s' % (n_reads, line1[1:])
      write('%s%s\n%s%s\n' % (line1, line2, line3, line4))
 
    if max_reads_in and n_reads >= max_reads_in:
      break
    
    line1 = readline()


  in_file_obj.close()
  
  if n_reads:
    n = float(n_reads)
  else:
    n = 1.0  
  
  mean_len /= n
  
  print(f'Input reads: {n_reads:,}\n' \
        f'Quality clipped: {n_qclip:,} ({100.0 * n_qclip/n:.2f}%)\n' \
        f'Too short: {n_short:,} ({100.0 * n_short/n:.2f}%)\n' \
        f'Mean read length: {mean_len:.2f}')
  
  
  # f'Tn5 ME: {n_jclip:,} ({100.0 * n_jclip/n:.2f}%)\n' \
    
  if adapt_seqs:
    print(f"3' adapter {n_adapt:,} ({100.0 * n_adapt/n:.2f}%)")

   

  move(clipped_file_temp, clipped_file)

  return clipped_file


def pair_mapped_hybrid_seqs(sam_file1, sam_file2, sam_file3, sam_file4, chromo_names,
                            file_root, samtools_exe, ambig=True, max_cis_sep=2000):
  
  
  paired_sam_file_name = tag_file_name(file_root, 'pair', '.sam')
  paired_bam_file_name = tag_file_name(file_root, 'pair', '.bam')
  paired_sam_file_name_temp = paired_sam_file_name + TEMP_EXT

  #if INTERRUPTED and os.path.exists(paired_sam_file_name) and not os.path.exists(paired_sam_file_name_temp):
  #  return paired_sam_file_name

  sam_file_obj = open(paired_sam_file_name_temp, 'w')
  
  # Add SAM header  
  
  sam_file_obj.write('@HD\tVN:1.1\tSO:unsorted\n')
  
  seq_heads = []
  prog_heads = []
  for sam_file in (sam_file1, sam_file3): # Only one for each build
    with open_file_r(sam_file) as file_obj:
      for line in file_obj:
        if line.startswith('@SQ'):
          seq_heads.append(line)
        elif line.startswith('@PG'):
          prog_heads.append(line)
          break
  
  for line in seq_heads:
    sam_file_obj.write(line)
 
  for line in prog_heads:
    sam_file_obj.write(line)
  
  cmd = ' '.join(sys.argv)
  sam_file_obj.write(f'@PG\tID:map_hybrid_reads\tPN:map_hybrid_reads\tVN:{VERSION}\tCL:"{cmd}"\n')  
  
  # - can copy most of head from one input
  # - but need all chromosome records, i.e. from hybrid/homologue
  
  write_sam_out = sam_file_obj.write

  file_objs = [open_file_r(sam_file) for sam_file in (sam_file1, sam_file2, sam_file3, sam_file4)]
  readlines = [f.readline for f in file_objs]
      
  n_map = [0,0,0,0]
  n_pairs = 0
  n_ambig = 0
  n_unpaired = 0
  n_unambig = 0
  n_unmapped = 0
  n_hybrid_ambig = 0
  n_hybrid_poor = 0
  n_hybrid_end_missing = 0
  n_primary_strand = [0,0,0,0]
  n_strand = [0,0,0,0]
  
  max_score = 0
  zero_ord = QUAL_ZERO_ORDS['phred33']
  really_bad_score = 2 * (max_score - 2 * (BOWTIE_MAX_AMBIG_SCORE_TOL+1) )
  close_second_best = 2 * BOWTIE_MAX_AMBIG_SCORE_TOL - 1
  
  # Go through same files and pair based on matching id
  # Write out any multi-position mapings

  # Skip to end of headers

  lines = [_skip_sam_header(readline) for readline in readlines]

  # Process data lines

  ids = [line[:ID_LEN] for line in lines]
  searchsorted = np.searchsorted

  while '' not in ids:
    _id = max(ids)
    
    unpaired = set()
    for i in range(4):
      while ids[i] and ids[i] < _id: # Reads may not match as some ends can be removed during clipping
        unpaired.add(ids[i])
        line = readlines[i]()
        lines[i] = line
        ids[i] = line[:ID_LEN]
      
    if unpaired:
      n_unpaired += len(unpaired)
      continue 
      
    else:
      sam_ends = [[], [], [], []]
      scores = [[], [], [], []]
      
      for i in range(4):
        j = 0
        
        while ids[i] == _id:
          n_map[i] += 1
          line = lines[i]
          data = line.split('\t')
          chromo = data[2]
          chr_name = chromo_names.get(chromo, chromo)
          
          if chromo != '*':
            revcomp = int(data[1]) & 0x10
            start = int(data[3])
            seq = data[9]
            qual = data[10]
            nbp = len(seq)
            end = start + nbp
            
            score, var = SCORE_TAG_SEARCH(line).groups()
            var_orig = var             
            score = int(score) # when --local : -nbp*2

            if revcomp and var[-1] == '0' and var[-2] in 'GCAT': # Ignore substitutions at the end e.g "C0"; add back subtracted score
              var = var[:-2]
            
              if seq[-1] != 'N': # No penalty for N's
                q = min(ord(qual[-1]) - zero_ord, 40.0)
                mp1 = 2 + floor(4*q/40.0)  # MX = 6, MN = 2.
                score = min(score+mp1, max_score)
                end -= 1

            elif var[0] == '0' and var[1] in 'GCAT':
              var = var[2:]
              
              if seq[0] != 'N':
                q = min(ord(qual[0]) - zero_ord, 40.0)
                mp2 = 2 + floor(4*q/40.0)  
                score = min(score+mp2, max_score)
                start += 1
           
            if revcomp:
              start, end = end, start  # The sequencing read started from the other end
            
            sam_ends[i].append((chr_name, start, line, score))
            scores[i].append(score)
            
            if j == 0:
              n_strand[i] += 1.0
 
              if not revcomp:
                n_primary_strand[i] += 1            
            
          line = readlines[i]()
          lines[i] = line
          ids[i] = line[:ID_LEN]
          j += 1
          
      for a in (0,1,2,3):
        if len(scores[a]) > 1:
          score_lim = max(scores[a]) - close_second_best # Allow one mismatch or two consec
          idx = [i for i, score in enumerate(scores[a]) if score > score_lim]
          sam_ends[a] = [sam_ends[a][i] for i in idx]
          scores[a] = [scores[a][i] for i in idx]
          
      for a, b in ((0,1), (2,3)):
        if (len(scores[a]) > 1) or (len(scores[b]) > 1):
          n_best_a = scores[a].count(max_score) # Zero is best/perfect score in end-to-end mode, 2 * seq len in local
          n_best_b = scores[b].count(max_score)
          
          if n_best_a * n_best_b == 1: # Only one perfect pair
            i = scores[a].index(max_score)
            j = scores[b].index(max_score)
            
            sam_ends[a] = [sam_ends[a][i]]
            sam_ends[b] = [sam_ends[b][j]]
          
          else:
            for end in (a, b):
              if len(scores[end]) > 1:
                chr_name1, start1 = sam_ends[end][0][:2]
                chr_name2, start2 = sam_ends[end][1][:2]
 
                if (chr_name1 == chr_name2) and (abs(start2-start1) < CLOSE_AMBIG): # For position ambiguous v. close in cis almost certainly correct
                  i = scores[end].index(max(scores[end]))
                  sam_ends[end] = [sam_ends[end][i]]
              
      n_pairs += 1
      iid = int(_id)
      
      if n_pairs % 10000 == 0:
        sys.stdout.write(f'\r .. {n_pairs:,}') 
        sys.stdout.flush()
      
      n0 = len(sam_ends[0])
      n1 = len(sam_ends[1])
      n2 = len(sam_ends[2])
      n3 = len(sam_ends[3])
      
      if (n0+n2) * (n1+n3) == 0: # One end or both ends have no mappings
        n_unmapped += 1
        continue
      
      elif min(n0, n1, n2, n3) == 0: # Not all ends accounted for # Only applicable for strains of same species
        n_hybrid_end_missing += 1
        continue      

      pairs = []
      for end_a, end_b in ((0,1), (2,3)): # , (0,3), (2,1)): # A:A, B:B, A:B, B:A genome pairings
        for i, (chr1, start1, line1, score1) in enumerate(sam_ends[end_a]):
          for j, (chr2, start2, line2, score2) in enumerate(sam_ends[end_b]):
             
            if chr1 != chr2:
              continue
                
            elif abs(start2-start1) > max_cis_sep:
              continue
            
            pairs.append((score1 + score2, i, j, line1, line2))            
      
      if not pairs:
        n_unmapped += 1
        continue
      
      pairs.sort(reverse=True)
      best_score = pairs[0][0]      

      if best_score < really_bad_score: # Nothing any good
        n_hybrid_poor += 1
        continue

      else:
        pairs = [x for x in pairs if x[0] >= best_score-BOWTIE_MAX_AMBIG_SCORE_TOL]
                
      #ambig_code = float(len(pairs))
      is_pos_ambig = False
      
      for score, i, j, line1, line2 in pairs:
        # See:  https://samtools.github.io/hts-specs/SAMv1.pdf
        # Set Paired in FLAG
        # Insert RNEXT : chromo name
        # Insert PNEXT : start seq position
        
        """
        line1 = _insert_mate_pair(line1, line2)
        line2 = _insert_mate_pair(line2, line1)
        
        """
        
        write_sam_out(line1)
        write_sam_out(line2)
        
        if i > 0 or j > 0:
          is_pos_ambig = True
        
      if is_pos_ambig:
        n_ambig += 1
          
      elif len(pairs) > 1: # Ambiguity only due to hybrid genome
        n_hybrid_ambig += 1
      
      else:
        n_unambig += 1
  
  info(f' .. {n_pairs:,}')
  
  if n_pairs:
    n = float(n_pairs)
  else:
    n = 1.0  
  
  print(f'Input read read pairs {n_pairs:,}')
  print(f'Unambiguous read pairs : {n_unambig:,} ({100.0 *  n_unambig/n:.2f}%)')
  print(f'Ambiguous read pairs : {n_ambig:,} ({100.0 *  n_ambig/n:.2f}%)')
  print(f'Hybrid-ambig read pairs : {n_hybrid_ambig:,} ({100.0 * n_hybrid_ambig /n:.2f}%)')
  
  print(f'Unmappable end : {n_unmapped:,} ({100.0 *  n_unmapped/n:.2f}%)')
  print(f'Missing mapping : {n_hybrid_end_missing:,} ({100.0 *  n_hybrid_end_missing/n:.2f}%)')
  print(f'Only bad mapping : {n_hybrid_poor:,} ({100.0 * n_hybrid_poor /n:.2f}%)')
  
  sam_file_obj.close()
  
  if samtools_exe:
    cmd_args = [samtools_exe, 'view', '-bS', paired_sam_file_name_temp]
    info(' '.join(cmd_args))
    proc = Popen(cmd_args, stderr=PIPE, stdout=open(paired_bam_file_name, 'wb'))
    proc.communicate()
    
    os.unlink(paired_sam_file_name_temp)

    return paired_bam_file_name
    
  else:
    move(paired_sam_file_name_temp, paired_sam_file_name)
  
    return paired_sam_file_name


def check_regular_file(file_path, critical=False):

  msg = ''
  is_ok = True

  if not os.path.exists(file_path):
    msg = 'File "%s" does not exist' % file_path
    is_ok = False

  elif not os.path.isfile(file_path):
    msg = 'Location "%s" is not a regular file' % file_path
    is_ok = False

  elif os.stat(file_path).st_size == 0:
    msg = 'File "%s" is of zero size ' % file_path
    is_ok = False

  elif not os.access(file_path, os.R_OK):
    msg = 'File "%s" is not readable' % file_path
    is_ok = False

  if critical and not is_ok:
    fatal(msg)

  return is_ok, msg


def check_index_file(file_path, sub_files=('.1', '.2', '.3', '.4', '.rev.1', '.rev.2'), critical=True):
     
  msg = ''
  is_ok = True

  if os.path.exists(file_path + '.1.bt2l'):
    file_ext = '.bt2l' # All build files should be long
  else:
    file_ext = '.bt2'

  for sub_file in sub_files:
    full_path = file_path + sub_file + file_ext
    is_ok, msg = check_regular_file(full_path)
    
    if not is_ok:
      msg = 'Genome index error. ' + msg    
      break

  if critical and not is_ok:
    fatal(msg)

  return is_ok, msg


def uncompress_gz_file(file_name):

  if file_name.endswith('.gz'):
    in_file_obj = gzip.open(file_name, 'rb')

    file_name = file_name[:-3]
    out_file_obj = open(file_name, 'w')
    write = out_file_obj.write
    
    for line in in_file_obj:
      write(line)

    # Faster alternative, given sufficient memory:
    # out_file_obj.write(infile_obj.read())

    in_file_obj.close()
    out_file_obj.close()

  return file_name
  
  
def index_genome(base_name, file_names, output_dir, indexer_exe='bowtie2-build',
                 table_size=10, quiet=True, pack=True, num_cpu=2):

  fasta_files = []
  for file_name in file_names:
    file_name = uncompress_gz_file(file_name)
    fasta_files.append(file_name)

  fasta_file_str = ','.join(fasta_files)

  cmd_args = [indexer_exe, '-f']

  if quiet:
    cmd_args.append('-q')

  if pack:
    cmd_args.append('-p')

  cmd_args += ['-t', str(table_size), '--threads', str(num_cpu), fasta_file_str, base_name]
  #cmd_args += ['-t', str(table_size), fasta_file_str, base_name]

  call(cmd_args, cwd=output_dir)

FILENAME_SPLIT_PATT = re.compile('[_\.]')

def merge_file_names(file_path1, file_path2, sep='_'):

  # same dir, need non truncated name

  dir_name1, file_name1 = os.path.split(file_path1)
  dir_name2, file_name2 = os.path.split(file_path2)

  if dir_name1 != dir_name2:
    msg = 'Attempt to merge file names for file from different directories'
    raise Exception(msg)

  file_root1, file_ext1 = os.path.splitext(file_name1)
  file_root2, file_ext2 = os.path.splitext(file_name2)

  if file_ext1 != file_ext2:
    msg = 'Attempt to merge file names with different file extensions'
    raise Exception(msg)

  parts1 = FILENAME_SPLIT_PATT.split(file_root1)
  parts2 = FILENAME_SPLIT_PATT.split(file_root2)
  parts3 = []

  n1 = len(parts1)
  n2 = len(parts2)
  n = max(n1, n2)

  for i in range(n):

    if (i < n1) and (i < n2):
      a = parts1[i]
      b = parts2[i]

      parts3.append(a)
      if a != b:
        parts3.append(b)

    elif i < n1:
      parts3.append(parts1[i])
    else:
      parts3.append(parts2[i])

  file_root3 = sep.join(parts3)

  file_path3 = os.path.join(dir_name1, file_root3 + file_ext1)
  
  return file_path3


FASTQ_READ_CHUNK = 1048576

def get_fastq_qual_scheme(file_path):
  """
  Guess the quality scoring scheme for a FASTQ file
  """

  file_obj = open_file_r(file_path, complete=False)

  lines = file_obj.readlines(FASTQ_READ_CHUNK)

  while lines[0][0] != '@': # Just in case of headers or other nonsense
    lines.pop(0)

  n_reads = 0
  quals = set()

  n = len(lines)
  n_reads += n // 4

  for i in range(n_reads):
    line_quals = lines[i*4+3][:-1]
    quals.update(set(line_quals))

  file_obj.close()

  quals = [ord(x) for x in quals]
  min_qual = min(quals)
  max_qual = max(quals)

  if min_qual < 33:
    scheme = 'integer'

  elif (max_qual < 75) and (min_qual < 59): # Sanger, Illumina 1.8+
    scheme = 'phred33'

  elif (max_qual > 74):
    if min_qual < 64:
      scheme = 'solexa'
    else:
      scheme = 'phred64'

  else:
    warn('FASTQ quality scheme could not be determined. Assuming Phred+33 (Illumina 1.8+)')
    scheme = 'phred33'

  return scheme


def map_reads(fastq_file, genome_index, align_exe, num_cpu, ambig, qual_scheme, job):

  sam_file_path = tag_file_name(fastq_file, 'map%d' % job, '.sam')
  sam_file_path_temp = sam_file_path + TEMP_EXT

  if os.path.exists(sam_file_path) and not os.path.exists(sam_file_path_temp):
    print(f'Using existing SAM file {sam_file_path}')
    return sam_file_path

  patt_1 = re.compile('(\d+) reads; of these:')
  patt_2 = re.compile('(\d+) \(.+\) aligned exactly 1 time')
  patt_3 = re.compile('(\d+) \(.+\) aligned 0 times')
  patt_4 = re.compile('(\d+) \(.+\) aligned >1 times')
  
  cmd_args = [align_exe,
              '-D', '20', '-R', '3', '-N', '0',  '-L', '20',  '-i', 'S,1,0.5', # similar to very-sensitive
              '-x', genome_index,
              '-k', '2',
              '--reorder',
              '--score-min', 'L,-0.6,-0.6',
              '-p', str(num_cpu),
              '-U', fastq_file,
              '--np', '0', # Penalty for N's
               '-S', sam_file_path_temp]

  if qual_scheme == 'phred33':
    cmd_args += ['--phred33']

  elif qual_scheme == 'phred64':
    cmd_args += ['--phred64']

  elif qual_scheme == 'solexa':
    cmd_args += ['--solexa-quals']

  n_reads = 0
  n_uniq = 0
  n_unmap = 0
  n_ambig = 0
  
  proc = Popen(cmd_args, stdin=PIPE, stderr=PIPE)
  std_out, std_err = proc.communicate()
  
  if std_err:
    std_err = std_err.decode('ascii')

    if 'Error' in std_err:
      warn(std_err)

    for line in std_err.split('\n'):

      if line.strip():

        match = patt_1.search(line)
        if match:
          n_reads = int(match.group(1))

        match = patt_2.search(line)
        if match:
          n_uniq = int(match.group(1))

        match = patt_3.search(line)
        if match:
          n_unmap = int(match.group(1))

        match = patt_4.search(line)
        if match:
          n_ambig = int(match.group(1))
  
  if n_reads:
    n  = float(n_reads)
  else:
    n = 1.0
  
  print(f'Input reads: {n_reads:,}\n' \
        f'Unique: {n_uniq:,} ({100.0 * n_uniq/n:.2f}%)\n' \
        f'Ambiguous: {n_ambig:,} ({100.0 * n_ambig/n:.2f}%)\n' \
        f'Unmapped: {n_unmap:,} ({100.0 * n_unmap/n:.2f}%)')  
  
  move(sam_file_path_temp, sam_file_path)

  return sam_file_path


def map_hybrid_reads(fastq_paths, genome_index1, genome_index2, chromo_name_files,
                     num_cpu=1, out_file=None, align_exe=None, samtools_exe=None, qual_scheme=None, min_qual=30,
                     adapt_seqs=None, trim_5=0, trim_3=0, max_cis_sep=2000, keep_files=False,
                     ambig=True, read_limit=None):
 
  if not align_exe:
    try: # Python version >= 3.3
      align_exe = which('bowtie2')

    except AttributeError:
      cmd_args = ['which', 'bowtie2']
      proc = Popen(cmd_args, stdin=PIPE, stdout=PIPE)
      align_exe, std_err_data = proc.communicate()
      align_exe = align_exe.strip()

  if not samtools_exe:
    try: 
      samtools_exe = which('samtools_exe')

    except AttributeError:
      cmd_args = ['which', 'samtools_exe']
      proc = Popen(cmd_args, stdin=PIPE, stdout=PIPE)
      samtools_exe, std_err_data = proc.communicate()
      samtools_exe = samtools_exe.strip()

  if fastq_paths:
    if len(fastq_paths) == 1:
      fatal('Only one FASTQ file specified (exactly two required)')

    if len(fastq_paths) > 2:
      fatal('More than two FASTQ files specified (exactly two required)')

  if not qual_scheme:
    qual_scheme = get_fastq_qual_scheme(fastq_paths[0])
      
  check_index_file(genome_index1)
  check_index_file(genome_index2)

  # Get base file name for output
  if out_file:
    file_root = os.path.splitext(out_file)[0]

  else:
    file_paths = []
    for fastq_path in fastq_paths:
      if fastq_path.lower().endswith('.gz'):
        fastq_path = fastq_path[:-3]

      file_paths.append(fastq_path)

    merged_path = merge_file_names(file_paths[0], file_paths[1])
    file_root = os.path.splitext(merged_path)[0]

  intermed_dir = file_root + '_nuc_processing_files'
  intermed_file_root = os.path.join(intermed_dir, os.path.basename(file_root))
  if not os.path.exists(intermed_dir):
    os.mkdir(intermed_dir)
  
  info(f'Clipping {fastq_paths[0]}')
  clipped_file1 = clip_reads(fastq_paths[0], intermed_file_root, qual_scheme,
                             min_qual, read_limit, adapt_seqs, trim_5, trim_3)
                             
  info(f'Clipping {fastq_paths[1]}')
  clipped_file2 = clip_reads(fastq_paths[1], intermed_file_root, qual_scheme,
                             min_qual, read_limit, adapt_seqs, trim_5, trim_3, is_second=True)

  info(f'Mapping FASTQ reads to first genome reference : {genome_index1} ...')
  sam_file1 = map_reads(clipped_file1, genome_index1, align_exe, num_cpu, ambig, qual_scheme, 1)
  sam_file2 = map_reads(clipped_file2, genome_index1, align_exe, num_cpu, ambig, qual_scheme, 2)

  info(f'Mapping FASTQ reads to second genome reference : {genome_index2} ...')
  sam_file3 = map_reads(clipped_file1, genome_index2, align_exe, num_cpu, ambig, qual_scheme, 3)
  sam_file4 = map_reads(clipped_file2, genome_index2, align_exe, num_cpu, ambig, qual_scheme, 4)
  
  #if not keep_files:
  #  for file_path in (clipped_file1, clipped_file2):
  #    if os.path.exists(file_path):
  #      os.unlink(file_path)

  chromo_names = {}
  for chromo_file in chromo_name_files:
    with open(chromo_file) as fp:
      for line in fp:
        line = line.strip()
        key, value = line.split()
        chromo_names[key] = value

  info(f'Pairing ambiguously mapped reads')
  paired_sam_file_name  = pair_mapped_hybrid_seqs(sam_file1, sam_file2, sam_file3, sam_file4, chromo_names,
                                                  file_root, samtools_exe, ambig=ambig, max_cis_sep=max_cis_sep)

  print(f'Wrote {paired_sam_file_name}')
  
  
def test_pair_mapped_hybrid_seqs():

  sam_directory = '/media/NAS/EDL1-DATA/delphi/scratch/wb104/nuc_process_runs/Hi-C_67_SLX-22381/P120F6_nuc_processing_files'
  sam_file_stub = 'P120F6_reads%d_clipped_map%d.sam'
  sam_files = [sam_file_stub % (2-n%2, n) for n in range(1, 5)]
  sam_file1, sam_file2, sam_file3, sam_file4 = [os.path.join(sam_directory, sam_file) for sam_file in sam_files]

  chromo_directory = '/media/NAS/EDL1-DATA/delphi/shared/genome_builds'
  chromo_files = ['mm_129_SvImJ_v2_chr_names.tsv', 'mm_CAST_EiJ_v2_chr_names.tsv']
  chromo_files = [os.path.join(chromo_directory, chromo_file) for chromo_file in chromo_files]

  file_root = '/media/NAS/EDL1-DATA/delphi/scratch/wb104/pair_hybrid_reads_code/out'

  chromo_names = {}
  for chromo_file in chromo_files:
    with open(chromo_file) as fp:
      for line in fp:
        line = line.strip()
        key, value = line.split()
        chromo_names[key] = value

  paired_sam_file_name = pair_mapped_hybrid_seqs(sam_file1, sam_file2, sam_file3, sam_file4, chromo_names,
                                                 file_root, ambig=True, max_cis_sep=20000)

  print(paired_sam_file_name)


def test_map_hybrid_reads():
  
  fastq_paths = ['/data1/ATAC_seq/SLX-22215.i701_i505.HLJJCDRX2.s_2.r_1.fq.gz',
                 '/data1/ATAC_seq/SLX-22215.i701_i505.HLJJCDRX2.s_2.r_2.fq.gz']
                 
  genome_index1 = '/data1/genome_builds/Mouse_EDL_B6_v2_chr'
  genome_index2 = '/data1/genome_builds/Mouse_EDL_CAST_v2_chr'
  
  chromo_name_files = ['/data1/genome_builds/Mouse_EDL_B6_v2_chr_names.tsv',
                       '/data1/genome_builds/Mouse_EDL_CAST_v2_chr_names.tsv']

  map_hybrid_reads(fastq_paths, genome_index1, genome_index2, chromo_name_files,
                   num_cpu=16, out_file=None, read_limit=None)
                   
if __name__ == '__main__':
  
  # test_pair_mapped_hybrid_seqs()

 test_map_hybrid_reads()
