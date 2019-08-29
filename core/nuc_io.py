import fnmatch
import gzip
import os
import re
import subprocess
import numpy as np
import glob

from io import BufferedReader, BufferedWriter

import core.nuc_util as util

# #   Globals  # #

FILENAME_SPLIT   = re.compile('[_\.]')
FILE_BUFFER_SIZE = 2**16
GZIP_EXTENSIONS = ('.gz','.gzip')

# #   Path naming  # #

def get_temp_path(file_path):
  '''Get a temporary path based on some other path or directory'''
  
  path_root, file_ext = os.path.splitext(file_path)
  
  return '%s_temp_%s%s' % (path_root, util.get_rand_string(8), file_ext)


def tag_file_name(file_path, tag, file_ext=None):

  dir_path, file_name = os.path.split(file_path)
  file_root, file_ext_old = os.path.splitext(file_name)

  if file_ext_old in GZIP_EXTENSIONS:
    file_root, file_ext_2 = os.path.splitext(file_root)
    file_ext_old = file_ext_2+file_ext_old
 
  if not file_ext:
    file_ext = file_ext_old
  
  file_name = '%s_%s%s' % (file_root, tag, file_ext)
  file_path = os.path.join(dir_path, file_name)

  return file_path 
   

def tag_file_path(file_path, new_ending):
  
  file_root, file_ext = os.path.splitext(file_path)

  if file_ext.lower() in GZIP_EXTENSIONS:
    file_root, file_ext = os.path.splitext(file_root)
      
  new_file_path = '%s_%s' % (file_root, new_ending)
  
  return new_file_path 

   
def get_file_ext(file_path):
  
  file_root, file_ext = os.path.splitext(file_path)
   
  if file_ext.lower() in GZIP_EXTENSIONS:
    file_root, file_ext = os.path.splitext(file_root)
   
  return file_ext


def get_file_root(file_path):
  
  return  os.path.splitext(os.path.basename(file_path))[0]


def get_safe_file_path(path_name, file_name=None):
   
  if file_name:
    file_path = os.path.join(path_name, file_name)
  else:
    file_path = path_name
 
  if os.path.exists(file_path):
    util.warn("%s already exists and won't be overwritten..." % file_path)
    
    path_root, file_ext = os.path.splitext(file_path)
    file_path = '%s_%s%s' % (path_root, util.get_rand_string(8), file_ext)
    
    util.info('Results will be saved in %s' % file_path)
  
  return file_path


def get_out_job_file_path(ref_file_path, format_str, insert_vals):
   
  dir_path = dirname(ref_data_path)
  globs = ['*'] + len(insert_vals)
  
  job_num = 1
  while glob.glob(os.path.join(dir_path, format_str.format(job, *globs))):
    job_num += 1
  
  file_name = format_str.format(job_num, *insert_vals)
  
  out_path = os.path.join(dir_path, file_name)  
  
  return
  

def check_file_ext(file_path, ext):
  
  if ext[0] != '.':
    ext = '.' + ext
  
  file_root, file_ext = os.path.splitext(file_path)
  
  if file_ext.lower() != ext.lower():
    file_path = file_root + ext
    
  return file_path
  
  
def is_ncc(file_path):

  if file_path.lower().endswith('.ncc') or file_path.lower().endswith('.ncc.gz'):
    return True
  else:
    return False
   
  
def merge_file_names(file_path1, file_path2, sep='_', prefix='', suffix=''):

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
  
  file_root3 = prefix + sep.join(parts3) + suffix
  
  file_path3 = os.path.join(dir_name1, file_root3 + file_ext1)
  
  return file_path3
 

def check_file_labels(file_labels, file_paths):
  
  file_names = [os.path.basename(x) for x in file_paths]
  
  if file_labels:
 
    while len(file_labels) < len(file_names):
      i = len(file_labels)
      file_labels.append(get_file_root(file_names[i]))
 
  else:
    file_labels = [get_file_root(x) for x in file_names]
    
  for i, label in enumerate(file_labels):
    file_labels[i] = file_labels[i].replace('_',' ')
  
  return file_labels
  

# #   Path operations  # # 

def match_files(file_paths, file_pattern):
  
  # Like glob, but on a list of strings
  
  return [fp for fp in file_paths if fnmatch.fnmatch(os.path.basename(fp), file_pattern)]


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
    in_file_obj = open_file(file_name)
 
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

    
def open_file(file_path, mode=None, buffer_size=FILE_BUFFER_SIZE, gzip_exts=('.gz','.gzip'), partial=False):
  """
  GZIP agnostic file opening
  """
  
  if os.path.splitext(file_path)[1].lower() in gzip_exts:
    if mode and 'w' in mode:
      file_obj = BufferedWriter(gzip.open(file_path, mode), buffer_size)
    else:
      if partial:
        file_obj = BufferedReader(gzip.open(file_path, mode or 'rt'), buffer_size)
        
      else:
        try:
          file_obj = subprocess.Popen(['zcat', file_path], stdout=subprocess.PIPE).stdout
        except OSError:
          file_obj = BufferedReader(gzip.open(file_path, mode or 'rt'), buffer_size)
 
  else:
    file_obj = open(file_path, mode or 'rU', buffer_size)
  
  return file_obj
 
 
def check_regular_file(file_path):
  
  msg = check_invalid_file(file_path)
  
  if msg:
    return False, msg
  
  return True, ''


def check_invalid_file(file_path, critical=True):
  
  msg = ''
  
  if not os.path.exists(file_path):
    msg = 'File "%s" does not exist' % file_path
 
  elif not os.path.isfile(file_path):
    msg = 'Location "%s" is not a regular file' % file_path
  
  elif os.stat(file_path).st_size == 0:
    msg = 'File "%s" is of zero size ' % file_path
    
  elif not os.access(file_path, os.R_OK):
    msg = 'File "%s" is not readable' % file_path

  if msg and critical:
    util.critical(msg)
  
  return msg
  

def is_same_file(file_path_a, file_path_b):
  
  file_path_a = os.path.abspath(file_path_a)
  file_path_b = os.path.abspath(file_path_b)
  
  return file_path_a == file_path_b
  
