import os

from nuc_tools import io, util

# #   Globals  # #

FASTQ_READ_CHUNK = 1048576


# #  FASTQ  format  # #

def check_format(file_path):
  
  file_obj = io.open_file(file_path, partial=True)
    
  lines = file_obj.readlines(FASTQ_READ_CHUNK)
  lines = [l for l in lines if l.rstrip()]
  
  for i in range(0, len(lines), 4):
    if not (lines[i][0] == '@') and (lines[i+2][0] == '+'):
      msg = 'File "%s" does not appear to be in FASTQ format'
      return False, msg % file_path
  
  return True, ''
  
  
def get_qual_scheme(file_path):
  """
  Guess the quality scoring scheme for a FASTQ file
  """
  
  file_obj = io.open_file(file_path, partial=True)
  
  lines = file_obj.readlines(FASTQ_READ_CHUNK)
  
  if not lines:
    util.critical('File "{}" contains no data'.format(file_path))
    
  while (not lines[0]) or (lines[0][0] != '@'): # Just in case of headers or other nonsense
    if not lines[0]:
      util.warn('Blank like skipped near top of file "{}"'.format(file_path))
      
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
    util.warn('FASTQ quality scheme could not be determined. Assuming Phred+33 (Illumina 1.8+)')
    scheme = 'phred33'

  return scheme


def pair_files(fastq_paths, pair_tags=('r_1','r_2'), err_msg='Could not pair FASTQ read files.'):
  
  if len(fastq_paths) != len(set(fastq_paths)):
    msg = '%s Repeat file path present.'
    util.critical(msg % (err_msg))
      
  t1, t2 = pair_tags
  
  paths_1 = []
  paths_2 = []
  
  for path in fastq_paths:
    dir_name, base_name = os.path.split(path)
    
    if (t1 in base_name) and (t2 in base_name):
      msg = '%s Tags "%s" and "%s" are ambiguous in file %s'
      util.critical(msg % (err_msg, t1, t2, base_name))
    
    elif t1 in base_name:
      paths_1.append((path, dir_name, base_name))
    
    elif t2 in base_name:
      paths_2.append((path, dir_name, base_name))
     
    else:
      msg = '%s File name %s does not contain tag "%s" or "%s"'
      util.critical(msg % (err_msg, base_name, t1, t2))
  
  n1 = len(paths_1)
  n2 = len(paths_2)
  
  if n1 != n2:
    msg = '%s Number of read 1 (%d) and read 2 (%d) files do not match'
    util.critical(msg % (err_msg, n1, n2))
  
  fastq_paths_r1 = []
  fastq_paths_r2 = []
  
  for path_1, dir_name_1, file_1 in paths_1:
    seek_file = file_1.replace(t1, t2)
    found = []
    
    for path_2, dir_name_2, file_2 in paths_2:
      if dir_name_1 != dir_name_2:
        continue
    
      if file_2 == seek_file:
        found.append(path_2)
    
    if len(found) == 0:
      # No need to check unpaired read 2 files as these always result in an isolated read 1
      msg = '%s No read 2 file "%s" found to pair with %s'
      util.critical(msg % (err_msg, seek_file, path_1))
         
    else: 
      # Repeat Read 2 files not possible as repeats checked earlier
      fastq_paths_r1.append(path_1)
      fastq_paths_r2.append(found[0])
  
  return fastq_paths_r1, fastq_paths_r2


