# Use Bowtie 2 mapping c.f. NucProcess

import re
import string
import sys

from math import floor
from shutil import move

import numpy as np

SESSION_KEY = ''.join(np.random.choice(tuple(string.ascii_letters), 8))
TEMP_EXT = '_temp_%s' % (SESSION_KEY)
READ_BUFFER = 2**16
QUAL_ZERO_ORDS = {'phred33':33, 'phred64':64, 'solexa':64}
BOWTIE_MAX_AMBIG_SCORE_TOL = 5
ID_LEN = 10
SCORE_TAG = re.compile(r'\sAS:i:(\S+).+\sMD:Z:(\S+)')
SCORE_TAG_SEARCH = SCORE_TAG.search
CLOSE_AMBIG = 1000

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

  import io, subprocess, gzip
 
  if os.path.splitext(file_path)[1].lower() in gzip_exts:

    if complete:
      try:
        file_obj = subprocess.Popen(['zcat', file_path], stdout=subprocess.PIPE).stdout
      except OSError:
        file_obj = io.BufferedReader(gzip.open(file_path, 'rb'), buffer_size)
   
    else:
      file_obj = io.BufferedReader(gzip.open(file_path, 'rb'), buffer_size)
   
    if sys.version_info.major > 2:
      file_obj = io.TextIOWrapper(file_obj, encoding="utf-8")

  else:
    if sys.version_info.major > 2:
      file_obj = open(file_path, 'rU', buffer_size, encoding='utf-8')
    
    else:
      file_obj = open(file_path, 'rU', buffer_size)

  return file_obj

def _skip_sam_header(readline):

  line = readline()

  while line and line[0] == '@':
    line = readline()

  return line
  
  
def pair_sam_lines(line1, line2):
  """
  Convert from single to paired SAM 
  """

  read1 = line1.split('\t')
  read2 = line2.split('\t')

  """
  https://samtools.github.io/hts-specs/SAMv1.pdf
  ---
  1 0x1 template having multiple segments in sequencing
  2 0x2 each segment properly aligned according to the aligner
  4 0x4 segment unmapped
  8 0x8 next segment in the template unmapped
  16 0x10 SEQ being reverse complemented
  32 0x20 SEQ of the next segment in the template being reverse complemented
  64 0x40 the first segment in the template
  128 0x80 the last segment in the template
  256 0x100 secondary alignment
  512 0x200 not passing filters, such as platform/vendor quality controls
  1024 0x400 PCR or optical duplicate
  2048 0x800 supplementary alignment
  """

  bits1 = int(read1[1])
  bits2 = int(read2[1])

  # Is paired and aligned
  bits1 |= 0x1
  bits1 |= 0x2
  bits2 |= 0x1
  bits2 |= 0x2

  # Match mate strand across pair
  if bits1 & 0x10:
    bits2 |= 0x20

  if bits2 & 0x10:
    bits1 |= 0x20

  # Set 1st & 2nd in pair
  bits1 |= 0x40
  bits2 |= 0x80

  # Insert the modified bitwise flags into the reads
  read1[1] = bits1
  read2[1] = bits2

  # RNEXT
  if read1[2] == read2[2]:
    read1[6] = '='
    read2[6] = '='
  else:
    read1[6] = read2[2]
    read2[6] = read1[2]

  # PNEXT
  read1[7] = read2[3]
  read2[7] = read1[3]

  line1 = '\t'.join([str(t) for t in read1])
  line2 = '\t'.join([str(t) for t in read2])

  return line1, line2


def pair_mapped_hybrid_seqs(sam_file1, sam_file2, sam_file3, sam_file4, chromo_names,
                            file_root, ambig=True, max_cis_sep=2000):

  paired_sam_file_name = tag_file_name(file_root, 'pair', '.sam')
  paired_sam_file_name_temp = paired_sam_file_name + TEMP_EXT

  #if INTERRUPTED and os.path.exists(paired_sam_file_name) and not os.path.exists(paired_sam_file_name_temp):
  #  return paired_sam_file_name

  sam_file_obj = open(paired_sam_file_name_temp, 'w')
  
  # Add SAM header # # # # # # # # # 
  # - can copy most of head from one input
  # - but need all chromosome records, i.e. from hybrid/homologue
  
  write_sam_out = sam_file_obj.writepair_sam_lines

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
      
      n0 = len(sam_ends[0])
      n1 = len(sam_ends[1])
      n2 = len(sam_ends[2])
      n3 = len(sam_ends[3])
      
      if (n0+n2) * (n1+n3) == 0: # One end or both ends have no mappings
        n_unmapped += 1
        continue
      
      elif min(n0, n1, n2, n3) == 0: # Not all ends accounted for 
        n_hybrid_end_missing += 1
        continue      

      pairs = []
      for end_a, end_b in ((0,1), (2,3), (0,3), (2,1)): # A:A, B:B, A:B, B:A genome pairings
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
        # https://samtools.github.io/hts-specs/SAMv1.pdf
        # Set Paired in FLAG
        # Insert RNEXT : chromo name
        # Insert PNEXT : start seq position
        
        line1, line2 = pair_sam_lines(line1, line2)
        
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

  sam_file_obj.close()

  move(paired_sam_file_name_temp, paired_sam_file_name)
  
  return paired_sam_file_name

if __name__ == '__main__':

  import os

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
                          file_root, ambig=True, max_cis_sep=2000)

  print(paired_sam_file_name)

