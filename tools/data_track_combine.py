PROG_NAME = 'data_track_combine'
VERSION = '1.0.0'
DESCRIPTION = 'Combine data tracks in BED format according to intersections and differences etc.'

def points_region_interset(pos, regions, exclude=0):
  """
  Return an array of indices for points which either do (exclude=0) or do not
  (exclude=1) interest with ant array of regions 
  """
  
  sel_overlap = 1 - int(exclude)
  
  n = 0
  n_pos   = len(pos)
  n_reg   = len(regions)
  indices = np.empty(n_pos, int)
  order   = np.array(regions[:,0].argsort(), int)  
  
  for i in range(n_pos):
    
    if pos[i] < regions[order[0],0]:
      if not sel_overlap:
        indices[n] = i
        n += 1
      
      continue
      
    a = 0
    for k in range(n_reg):
      j = order[k]
      
      if (regions[j,0] <= pos[i]) and (pos[i] <= regions[j,1]):
        a = 1
        break
 
      if pos[i] < regions[j, 0]:
        break
        
    if sel_overlap == a:
      indices[n] = i
      n += 1
  
  return indices[:n]
  
  
def data_track_combine(bed_data_path, chromo_limits, out_dir, bin_size, smooth=False,
                                      out_bed_path=None, include_regions=None, exclude_regions=None, intersect_width=2000):
  """
  Create a binned data track, stored as a BED file, from an input BED file and specified chromsome
  limits. Saves data in a specified directory. Can include or exclude data points accoring to any
  overlap with lists of other regions.
  """
 
  from nuc_tools import util
 
  if not out_bed_path:
    file_root = os.path.splitext(os.path.basename(bed_data_path))[0]
    
    out_bed_path = '%s_%dkb_bin.bed' % (file_root, int(bin_size/1000))
    out_bed_path = os.path.join(out_dir, out_bed_path)
  
  region_dict, value_dict, label_dict = bed.load_bed_data_track(bed_data_path) 
  bin_region_dict = {}
  bin_value_dict = {}
  delta = bin_size-1
  hist_data = []
  half_bin = bin_size/2
  
  for chromo in sorted(region_dict):
    start, end = chromo_limits[chromo]
    
    regions = np.array(region_dict[chromo])
    values = value_dict[chromo]
    
    if include_regions:
      # Keep only data points which intersect these regions   
           
      for intersect_regions in include_regions:
        filter_regions = intersect_regions[chromo] + np.array([-intersect_width, intersect_width])
        idx = points_region_interset(regions.mean(axis=1), filter_regions)
        regions = regions[idx]
        values = values[idx]
    
    if exclude_regions:
      # Remove datapoints which intersect these regions
      
      for intersect_regions in exclude_regions:
        filter_regions = intersect_regions[chromo] + np.array([-intersect_width, intersect_width])
        idx = points_region_interset(regions.mean(axis=1), filter_regions, exclude=1)
        regions = regions[idx]
        values = values[idx]
      
    if smooth:
      # Widen the data point regions by half a bin width for smoothing
      d = (regions[:,1]-regions[:,0])/2
      regions[:,0] -= half_bin - d
      regions[:,1] += half_bin - d
    
    hist = bin_region_values(regions, values, bin_size, start, end)
    pos = np.arange(start, end, bin_size)
     
    bin_region_dict[chromo] = np.array([pos, pos+delta]).T
    bin_value_dict[chromo] = hist
    hist_data.append(hist)
  
  hist_data = np.concatenate(hist_data, axis=0)
  
  bed.save_bed_data_track(out_bed_path, bin_region_dict, bin_value_dict)
  util.info('  .. saved %s' % out_bed_path)
  
  return out_bed_path

def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(metavar='BED_FILE', nargs='+', dest='i',
                         help='Input data track file in BED format. Accepts wildcards.')

  args = vars(arg_parse.parse_args(argv))
                                
  data_path = args['i']
  
  data_track_combine(data_path, include_paths, exclude_paths, intersect_width)
  
if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()




