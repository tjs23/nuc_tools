import gzip
import numpy as np

# #   Globals  # #

#NCC_FORMAT       = '%s %d %d %d %d %s %s %d %d %d %d %s %d %d %d\n'


# #   Nuc Formats  # # 

def load_ncc_file(file_path):
  """Load chromosome and contact data from NCC format file, as output from NucProcess"""
  
  import nuc_io as io
  
  with io.open_file(file_path) as file_obj:
  
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


def load_n3d_coords(file_path):
  """
  Load genome structure coordinates and particle sequence positions from an N3D format file.
  
  Args: 
      file_path: str ; Location of N3D (text) format file

  Returns:
      dict {str:ndarray(n_coords, int)}                  ; {chromo: seq_pos_array}
      dict {str:ndarray((n_models, n_coords, 3), float)} ; {chromo: coord_3d_array}
  
  """  
  
  import nuc_io as io

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
        
        if chromo.lower()[:3] == 'chr':
          chromo = chromo[3:]
        
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
