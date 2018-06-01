import fnmatch
import gzip
import os
import re

import nuc_util

# #   Globals  # #

FILENAME_SPLIT   = re.compile('[_\.]')


# #   Path naming  # #

def get_temp_path(file_path):
  '''Get a temporary path based on some other path or directory'''
  
  path_root, file_ext = os.path.splitext(file_path)
  
  return '%s_temp_%s%s' % (path_root, nuc_util.get_rand_string(8), file_ext)

  
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
    nuc_util.warn("%s already exists and won't be overwritten..." % file_path)
    
    path_root, file_ext = os.path.splitext(file_path)
    file_path = '%s_%s%s' % (path_root, nuc_util.get_rand_string(8), file_ext)
    
    nuc_util.info('Results will be saved in %s' % file_path)
  
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


