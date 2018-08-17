import fnmatch
import gzip
import os
import re
import subprocess

from io import BufferedReader, BufferedWriter

import core.nuc_util as util

# #   Globals  # #

FILENAME_SPLIT   = re.compile('[_\.]')
FILE_BUFFER_SIZE = 2**16


# #   Path naming  # #

def get_temp_path(file_path):
  '''Get a temporary path based on some other path or directory'''
  
  path_root, file_ext = os.path.splitext(file_path)
  
  return '%s_temp_%s%s' % (path_root, util.get_rand_string(8), file_ext)


def tag_file_name(file_path, tag, file_ext=None):

  dir_path, file_name = os.path.split(file_path)

  if file_name.endswith('.gz'):
    file_root, file_ext_old = os.path.splitext(file_name[:-3])
    file_name = '%s_%s%s.gz' % (file_root, tag, (file_ext or file_ext_old))

  else:
    file_root, file_ext_old = os.path.splitext(file_name)
    file_name = '%s_%s%s' % (file_root, tag, (file_ext or file_ext_old))

  file_path = os.path.join(dir_path, file_name)

  return file_path 
   
   
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
    util.warn("%s already exists and won't be overwritten..." % file_path)
    
    path_root, file_ext = os.path.splitext(file_path)
    file_path = '%s_%s%s' % (path_root, util.get_rand_string(8), file_ext)
    
    util.info('Results will be saved in %s' % file_path)
  
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

    
def open_file(file_path, mode=None, buffer_size=FILE_BUFFER_SIZE, gzip_exts=('.gz','.gzip')):
  """
  GZIP agnostic file opening
  """
  
  if os.path.splitext(file_path)[1].lower() in gzip_exts:
    if mode and 'w' in mode:
      file_obj = BufferedWriter(gzip.open(file_path, mode), buffer_size)
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


def check_invalid_file(file_path):

  if not os.path.exists(file_path):
    msg = 'File "%s" does not exist'
    return msg % file_path
  
  if not os.path.isfile(file_path):
    msg = 'Location "%s" is not a regular file'
    return msg % file_path
  
  if os.stat(file_path).st_size == 0:
    msg = 'File "%s" is of zero size '
    return msg % file_path
    
  if not os.access(file_path, os.R_OK):
    msg = 'File "%s" is not readable'
    return msg % file_path

  return ''

def is_same_file(file_path_a, file_path_b):
  
  file_path_a = os.path.abspath(file_path_a)
  file_path_b = os.path.abspath(file_path_b)
  
  return file_path_a == file_path_b
  
