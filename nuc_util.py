import gzip
import multiprocessing
import numpy as np
import os
import random
import re
import string
import subprocess
import sys
import traceback
import uuid


# #   Globals  # #

QUIET            = False # Global verbosity flag
LOGGING          = False # Global file logging flag
MAX_CORES        = multiprocessing.cpu_count()
FASTQ_READ_CHUNK = 1048576
TEMP_ID          = '%s' % uuid.uuid4()
LOG_FILE_PATH    = 'nuc-tools-out-%s.log' % TEMP_ID
LOG_FILE_OBJ     = None # Created when needed
FILENAME_SPLIT   = re.compile('[_\.]')
NCC_FORMAT       = '%s %d %d %d %d %s %s %d %d %d %d %s %d %d %d\n'


# #   Srcreen reporting  # # 

def report(msg):
 
  if LOGGING:
    if not LOG_FILE_OBJ:
      LOG_FILE_OBJ = open(LOG_FILE_PATH, 'w')
      
    LOG_FILE_OBJ.write(msg)
  
  if not QUIET:
    print(msg)
   

def warn(msg, prefix='WARNING'):

  report('%s: %s' % (prefix, msg))

 
def critical(msg, prefix='FAILURE'):

  report('%s: %s' % (prefix, msg))
  sys.exit(0)


def info(msg, prefix='INFO'):

  report('%s: %s' % (prefix, msg))



# #   Path naming  # #

def get_temp_path(file_path):
  '''Get a temporary path based on some other path or directory'''
  
  path_root, file_ext = os.path.splitext(file_path)
  
  return '%s_temp_%s%s' % (path_root, get_rand_string(8), file_ext)

  
def get_file_ext(file_path):
  
  file_root, file_ext = os.path.splitext(file_path)
   
  if file_ext.lower() in ('.gz','.gzip'):
    file_root, file_ext = os.path.splitext(file_root)
   
  return file_ext


def get_safe_file_path(path_name, file_name=None):
   
  if file_name:
    file_path = os.path.join(path_name, file_name)
  else:
    file_path = path_name
 
  if os.path.exists(file_path):
    warn("%s already exists and won't be overwritten..." % file_path)
    
    path_root, file_ext = os.path.splitext(file_path)
    file_path = '%s_%s%s' % (path_root, get_rand_string(8), file_ext)
    
    info('Results will be saved in %s' % file_path)
  
  return file_path
  
  
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
  
  parts1 = FILENAME_SPLIT.split(file_root1)
  parts2 = FILENAME_SPLIT.split(file_root2)
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
 

# #   Path operations  # # 

def match_files(file_paths, file_pattern):

  from fnmatch import fnmatch
  from os.path import basename
  
  # Like glob, but on a list of strings
  
  return [fp for fp in file_paths if fnmatch(basename(fp), file_pattern)]


def makedirs(dir_path, exist_ok=False):
  
  try: # Python 3
    os.makedirs(dir_path, exist_ok=exist_ok)
  
  except TypeError:
    if not (os.path.exists(dir_path) and os.path.isdir(dir_path)):
      os.makedirs(dir_path)


# #  File operations  # #

def check_exe(file_name):
  
  if not os.path.exists(file_name):
    if not locate_exe(file_name):
      critical('Could not locate command exacutable "%s" in system $PATH' % file_name)
 

def locate_exe(file_name):
 
  for path in os.environ["PATH"].split(os.pathsep):
    if os.path.exists(os.path.join(path, file_name)):
      return os.path.join(path, file_name)

  return None
  
  
def uncompress_file(file_name):

  if file_name.endswith('.gz'):
    in_file_obj = gzip.open(file_name, 'rb')
 
    file_name = file_name[:-3]
    out_file_obj = open(file_name, 'w')

    for line in in_file_obj:
      out_file_obj.write(line)
    
    # Faster alternative, given sufficient memory:
    # out_file_obj.write(infile_obj.read())

    in_file_obj.close()
    out_file_obj.close()

  return file_name    


def compress_file(file_path):

  in_file_obj = open(file_path)
  out_file_path = file_path + '.gz'
  
  out_file_obj = gzip.open(out_file_path, 'wb')
  out_file_obj.writelines(in_file_obj)
  
  in_file_obj.close()
  out_file_obj.close()
  
  os.unlink(file_path)
  
  return out_file_path

    
def open_file(file_path, mode=None, gzip_exts=('.gz','.gzip')):
  """
  GZIP agnostic file opening
  """
  
  if os.path.splitext(file_path)[1].lower() in gzip_exts:
    file_obj = gzip.open(file_path, mode or 'rt')
  else:
    file_obj = open(file_path, mode or 'rU')
  
  return file_obj
 
 
def check_regular_file(file_path):

  msg = ''
  
  if not os.path.exists(file_path):
    msg = 'File "%s" does not exist'
    return False, msg % file_path
  
  if not os.path.isfile(file_path):
    msg = 'Location "%s" is not a regular file'
    return False, msg % file_path
  
  if os.stat(file_path).st_size == 0:
    msg = 'File "%s" is of zero size '
    return False, msg % file_path
    
  if not os.access(file_path, os.R_OK):
    msg = 'File "%s" is not readable'
    return False, msg % file_path
  
  return True, msg


# #  Job execution  # #    
    
    
def _parallel_func_wrapper(queue, target_func, proc_data, common_args):
  
  for t, data_item in proc_data:
    result = target_func(data_item, *common_args)
    
    if queue:
      queue.put((t, result))
  

def parallel_split_job(target_func, split_data, common_args, num_cpu=MAX_CORES, collect_output=True):
  
  num_tasks   = len(split_data)
  num_process = min(num_cpu, num_tasks)
  processes   = []
  
  if collect_output:
    queue = multiprocessing.Queue() # Queue will collect parallel process output
  
  else:
    queue = None
    
  for p in range(num_process):
    # Task IDs and data for each task
    # Each process can have multiple tasks if there are more tasks than processes/cpus
    proc_data = [(t, data_item) for t, data_item in enumerate(split_data) if t % num_cpu == p]
    args = (queue, target_func, proc_data, common_args)

    proc = multiprocessing.Process(target=_parallel_func_wrapper, args=args)
    processes.append(proc)
  
  for proc in processes:
    proc.start()
  
  
  if queue:
    results = [None] * num_tasks
    
    for i in range(num_tasks):
      t, result = queue.get() # Asynchronous fetch output: whichever process completes a task first
      results[t] = result
 
    queue.close()
 
    return results
  
  else:
    for proc in processes: # Asynchromous wait and no output captured
      proc.join()
    
    
def call(cmd_args, stdin=None, stdout=None, stderr=None, verbose=True, wait=True, path=None):
  """
  Wrapper for external calls to log and report commands,
  open stdin, stderr and stdout etc.
  """
  
  if verbose:
    info(' '.join(cmd_args))
  
  if path:
    env = dict(os.environ)
    prev = env.get('PATH', '')
    
    if path not in prev.split(':'):
      env['PATH'] = prev + ':' + path
  
  else:
    env = None # Current environment variables 
      
  if stdin and isinstance(stdin, str):
    stdin = open(stdin)

  if stdout and isinstance(stdout, str):
    stdout = open(stdout, 'w')

  if stderr and isinstance(stderr, str):
    stderr = open(stderr, 'a')
  
  if wait:
    subprocess.call(cmd_args, stdin=stdin, stdout=stdout, stderr=stderr, env=env)
      
  else:
    subprocess.Popen(cmd_args, stdin=stdin, stdout=stdout, stderr=stderr, env=env)
  

# #  Strings  # #

 
def get_rand_string(size):
  
  return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for i in range(size))


# #  FASTQ  format  # #

def check_fastq_file(file_path):
  
  file_obj = open_file(file_path)
    
  lines = file_obj.readlines(FASTQ_READ_CHUNK)
  lines = [l for l in lines if l.rstrip()]
  
  for i in range(0, len(lines), 4):
    if not (lines[i][0] == '@') and (lines[i+2][0] == '+'):
      msg = 'File "%s" does not appear to be in FASTQ format'
      return False, msg % file_path
  
  return True, ''
  
  
def get_fastq_qual_scheme(file_path):
  """
  Guess the quality scoring scheme for a FASTQ file
  """
  
  file_obj = open_file(file_path)
  
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
    scheme = 'integer-quals'
  
  elif (max_qual < 75) and (min_qual < 59): # Sanger, Illumina 1.8+
    scheme = 'phred33-quals'
  
  elif (max_qual > 74): 
    if min_qual < 64:
      scheme = 'solexa-quals'
    else:
      scheme = 'phred64-quals'
  
  else:
    warn('FASTQ quality scheme could not be determined. Assuming Phred+33 (Illumina 1.8+)')
    scheme = 'phred33-quals'

  return scheme


def pair_fastq_files(fastq_paths, pair_tags=('r_1','r_2'), err_msg='Could not pair FASTQ read files.'):
  
  if len(fastq_paths) != len(set(fastq_paths)):
    msg = '%s Repeat file path present.'
    critical(msg % (err_msg))
      
  t1, t2 = pair_tags
  
  paths_1 = []
  paths_2 = []
  
  for path in fastq_paths:
    dir_name, base_name = os.path.split(path)
    
    if (t1 in base_name) and (t2 in base_name):
      msg = '%s Tags "%s" and "%s" are ambiguous in file %s'
      critical(msg % (err_msg, t1, t2, base_name))
    
    elif t1 in base_name:
      paths_1.append((path, dir_name, base_name))
    
    elif t2 in base_name:
      paths_2.append((path, dir_name, base_name))
     
    else:
      msg = '%s File name %s does not contain tag "%s" or "%s"'
      critical(msg % (err_msg, base_name, t1, t2))
  
  n1 = len(paths_1)
  n2 = len(paths_2)
  
  if n1 != n2:
    msg = '%s Number of read 1 (%d) and read 2 (%d) files do not match'
    critical(msg % (err_msg, n1, n2))
  
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
      critical(msg % (err_msg, seek_file, path_1))
         
    else: 
      # Repeat Read 2 files not possible as repeats checked earlier
      fastq_paths_r1.append(path_1)
      fastq_paths_r2.append(found[0])
  
  return fastq_paths_r1, fastq_paths_r2


# #   BAM/SAM format  # #

def get_bam_chromo_sizes(bam_file_path):

  # Looks in header of BAM file to get chromosome/contig names and thier lengths  
    
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


# #   Nuc Formats  # # 

def load_ncc_file(file_path):
  """Load chromosome and contact data from NCC format file, as output from NucProcess"""
  
  if file_path.endswith('.gz'):
    import gzip
    file_obj = gzip.open(file_path)
  
  else:
    file_obj = open(file_path) 
  
  # Observations are treated individually in single-cell Hi-C,
  # i.e. no binning, so num_obs always 1 for each contact
  num_obs = 1  
    
  contact_dict = {}
  chromosomes = set()
    
  for line in file_obj:
    chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, ambig_group, pair_id, swap_pair = line.split()
    
    if strand_a == '+':
      pos_a = int(f_start_a)
    else:
      pos_a = int(f_end_a)
    
    if strand_b == '+':
      pos_b = int(f_start_b)       
    else:
      pos_b = int(f_end_b)
 
    if chr_a > chr_b:
      chr_a, chr_b = chr_b, chr_a
      pos_a, pos_b = pos_b, pos_a
    
    if chr_a not in contact_dict:
      contact_dict[chr_a] = {}
      chromosomes.add(chr_a)
      
    if chr_b not in contact_dict[chr_a]:
      contact_dict[chr_a][chr_b] = [] 
      chromosomes.add(chr_b)
        
    contact_dict[chr_a][chr_b].append((pos_a, pos_b, num_obs, int(ambig_group)))
   
  file_obj.close()
  
  chromo_limits = {}
    
  for chr_a in contact_dict:
    for chr_b in contact_dict[chr_a]:
      contacts = np.array(contact_dict[chr_a][chr_b]).T
      contact_dict[chr_a][chr_b] = contacts
      
      seq_pos_a = contacts[1]
      seq_pos_b = contacts[2]
      
      min_a = min(seq_pos_a)
      max_a = max(seq_pos_a)
      min_b = min(seq_pos_b)
      max_b = max(seq_pos_b)
        
      if chr_a in chromo_limits:
        prev_min, prev_max = chromo_limits[chr_a]
        chromo_limits[chr_a] = [min(prev_min, min_a), max(prev_max, max_a)]
      else:
        chromo_limits[chr_a] = [min_a, max_a]
      
      if chr_b in chromo_limits:
        prev_min, prev_max = chromo_limits[chr_b]
        chromo_limits[chr_b] = [min(prev_min, min_b), max(prev_max, max_b)]
      else:
        chromo_limits[chr_b] = [min_b, max_b]
         
  chromosomes = sorted(chromosomes)      
        
  return chromosomes, chromo_limits, contact_dict


def export_n3d_coords(file_path, coords_dict, seq_pos_dict):
  
  file_obj = open(file_path, 'w')
  write = file_obj.write
  
  for chromo in seq_pos_dict:
    chromo_coords = coords_dict[chromo]
    chromo_seq_pos = seq_pos_dict[chromo]
    
    num_models = len(chromo_coords)
    num_coords = len(chromo_seq_pos)
    
    line = '%s\t%d\t%d\n' % (chromo, num_coords, num_models)
    write(line)
    
    for j in range(num_coords):
      data = chromo_coords[:,j].ravel().tolist()
      data = '\t'.join('%.8f' % d for d in  data)
      
      line = '%d\t%s\n' % (chromo_seq_pos[j], data)
      write(line)

  file_obj.close()

