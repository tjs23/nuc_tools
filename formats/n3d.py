import numpy as np
import math

def get_chromo_limits(n3d_paths):
  """
  Get the sequence limits and bin size of particles for chromosomes
  in multiple structures.
  """
  
  chromo_limits = {}
  bin_size = None
  
  for n3d_coord_path in n3d_paths:
    seq_pos_dict, coords_dict = load_n3d_coords(n3d_coord_path)
  
    for chromo in seq_pos_dict:
      if not bin_size:
        a, b, = seq_pos_dict[chromo][:2]
        bin_size = b-a
 
      p1 = seq_pos_dict[chromo].min()
      p2 = seq_pos_dict[chromo].max()
      p1 = bin_size * int(p1/bin_size)
      p2 = bin_size * (1+int(math.ceil(p2/float(bin_size))))
 
      if chromo in chromo_limits:
        q1, q2 = chromo_limits[chromo]
        chromo_limits[chromo] = min(p1, q1), max(p2, q2)
      else:
        chromo_limits[chromo] = p1, p2
      
  return chromo_limits, bin_size
  
  
def export_n3d_coords(file_path, coords_dict, seq_pos_dict):
  
  return save_n3d_coords(file_path, coords_dict, seq_pos_dict)


def save_n3d_coords(file_path, coords_dict, seq_pos_dict):
  """
  Save genome structure coordinates and cooresponding particle sequence positions to an N3D format file.
  
  Args: 
      file_path: str ; Location to save N3D (text) format file
      coords_dict: {str:ndarray(n_coords, int)} ; {chromo: seq_pos_array}
      seq_pos_dict: {str:ndarray((n_models, n_coords, 3), float)} ; {chromo: coord_3d_array}

  
  """  
  
  file_obj = open(file_path, 'w')
  write = file_obj.write
  
  for chromo in seq_pos_dict:
    chromo_coords = coords_dict[chromo]
    chromo_seq_pos = seq_pos_dict[chromo]
    
    num_models = len(chromo_coords)
    num_coords = len(chromo_seq_pos)
    
    if chromo[:3].lower() != 'chr':
      chromo_name = 'chr' + chromo
    else:
      chromo_name = chromo
    
    line = '%s\t%d\t%d\n' % (chromo_name, num_coords, num_models)
    write(line)
    
    for j in range(num_coords):
      data = chromo_coords[:,j].ravel().tolist()
      data = '\t'.join('%.8f' % d for d in  data)
      
      line = '%d\t%s\n' % (chromo_seq_pos[j], data)
      write(line)

  file_obj.close()


def get_bin_size(file_path):
  
  import nuc_tools.core.nuc_io as io
  
  bin_size = None
  
  with io.open_file(file_path) as file_obj:
    chromo = None
    prev_seq_pos = None
    
    for line in file_obj:
      data = line.split()
      n_items = len(data)
 
      if not n_items:
        continue
 
      elif data[0] == '#':
        continue
 
      elif n_items == 3:
        chromo, n_coords, n_models = data
        check = (int(n_models) * 3) + 1
        continue
        
      elif not chromo:
        raise Exception('Missing chromosome record in file %s' % file_path)
 
      elif n_items != check:
        msg = 'Data size in file %s does not match Position + Models * Positions * 3'
        raise Exception(msg % file_path)
 
      if prev_seq_pos:
        bin_size =  int(data[0]) - prev_seq_pos
        break
      else:
        prev_seq_pos = int(data[0])
  
  return bin_size
  

def load_n3d_coords(file_path):
  """
  Load genome structure coordinates and particle sequence positions from an N3D format file.
  
  Args: 
      file_path: str ; Location of N3D (text) format file

  Returns:
      dict {str:ndarray(n_coords, int)}                  ; {chromo: seq_pos_array}
      dict {str:ndarray((n_models, n_coords, 3), float)} ; {chromo: coord_3d_array}
  
  """  
  
  import core.nuc_io as io

  seq_pos_dict = {}
  coords_dict = {}  
  
  with io.open_file(file_path) as file_obj:
    chromo = None
 
    for line in file_obj:
 
      data = line.split()
      n_items = len(data)
 
      if not n_items:
        continue
 
      elif data[0] == '#':
        continue
 
      elif n_items == 3:
        chromo, n_coords, n_models = data
        
        #if chromo.lower()[:3] == 'chr':
        #  chromo = chromo[3:]
        
        if chromo in coords_dict:
          raise Exception('Duplicate chromosome "%s" records in file %s' % (chromo, file_path))
        
        n_coords = int(n_coords)
        n_models = int(n_models)
 
        chromo_seq_pos = np.empty(n_coords, int)
        chromo_coords = np.empty((n_models, n_coords, 3), float)
 
        coords_dict[chromo]  = chromo_coords
        seq_pos_dict[chromo] = chromo_seq_pos
 
        check = (n_models * 3) + 1
        i = 0
 
      elif not chromo:
        raise Exception('Missing chromosome record in file %s' % file_path)
 
      elif n_items != check:
        msg = 'Data size in file %s does not match Position + Models * Positions * 3'
        raise Exception(msg % file_path)
 
      else:
        chromo_seq_pos[i] = int(data[0])
 
        coord = [float(x) for x in data[1:]]
        coord = np.array(coord).reshape(n_models, 3)
        chromo_coords[:,i] = coord
        i += 1
 
  return seq_pos_dict, coords_dict
