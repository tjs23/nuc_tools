import os, sys
import numpy as np
from os.path import dirname

PROG_NAME = 'data_track_filter'
VERSION = '1.0.0'
DESCRIPTION = 'Filter data tracks in BED format according to quantile values and/or intersections and differences to other data tracks'
DEFAULT_QUANTILE_MIN = 0.0
DEFAULT_QUANTILE_MAX = 100.0
DEFAULT_SEP = 10000

  
def data_track_filter(bed_data_path, out_bed_path, include_paths, exclude_paths, quantile_lower=DEFAULT_QUANTILE_MAX,
                      quantile_upper=DEFAULT_QUANTILE_MIN, include_seps=DEFAULT_SEP, exclude_seps=DEFAULT_SEP,
                      include_any=False, exclude_any=False):

  from nuc_tools import util
  from formats import bed
  
  if not out_bed_path:
    file_root = os.path.splitext(bed_data_path)[0]
    out_bed_path = '%s_filtered.bed' % file_root
  
  util.info('Loading data tracks')
  
  region_dict, value_dict, label_dict = bed.load_bed_data_track(bed_data_path)
  
  include_regions = [bed.load_bed_data_track(pth)[0] for pth in include_paths]
  exclude_regions = [bed.load_bed_data_track(pth)[0] for pth in exclude_paths]
  
  if len(include_seps) == 1:
    val = include_seps[0]
    include_seps = [val] * len(include_regions)

  if len(exclude_seps) == 1:
    val = exclude_seps[0]
    exclude_seps = [val] * len(exclude_regions)
  
  filt_region_dict = {}
  filt_value_dict = {}
  filt_label_dict = {}
  
  if quantile_lower > 0.0 or quantile_upper < 100.0:
    lower_limit, upper_limit = np.percentile(np.concatenate([value_dict[c] for c in value_dict]), (quantile_lower, quantile_upper))
  else:
    lower_limit = None
    upper_limit = None
  
  util.info('Filtering regions')
  n = 0
  m = 0
  
  for chromo in sorted(region_dict):
    regions = np.array(region_dict[chromo])
    labels = label_dict[chromo]
    centers = regions.mean(axis=1)
    values = value_dict[chromo]
    n0 = len(values)
    n += n0
        
    if label_dict:
      labels = label_dict[chromo]    
    
    if lower_limit or upper_limit:
      keep = (lower_limit <= values) & (values <= upper_limit)
      regions = regions[keep]
      centers = centers[keep]
      values = values[keep]
    
      if label_dict:
        labels = [x for i, x in enumerate(labels) if keep[i]]
        
    if include_regions and len(values):
      # Keep only data points which intersect these regions   
      keep = np.zeros(len(centers))
      
      for i, intersect_regions in enumerate(include_regions):
        if not len(intersect_regions[chromo]):
          continue
          
        width = include_seps[i]
        filter_regions = intersect_regions[chromo] + np.array([-width, width])
        
        s = intersect_regions[chromo][:,0]
        e = intersect_regions[chromo][:,0]
        idx = s.argsort()
        
        f = util.points_region_intersect(centers, filter_regions)
        
        keep += f.astype(int)
        
      if include_any: # Any track keeps
        keep = keep > 0
      
      else: # All tracks
        keep = keep == len(include_regions)
      
      regions = regions[keep]
      centers = centers[keep]
      values = values[keep]
      
      if label_dict:
        labels = [x for i, x in enumerate(labels) if keep[i]]
         
    if exclude_regions and len(values):
      # Remove datapoints which intersect these regions
      excl = np.zeros(len(centers))
      
      for i, intersect_regions in enumerate(exclude_regions):
        if not len(intersect_regions[chromo]):
          continue

        width = exclude_seps[i]
        filter_regions = intersect_regions[chromo] + np.array([-width, width])
        excl += util.points_region_intersect(centers, filter_regions).astype(int)
      
      if exclude_any: # Excluded by any 
        excl = excl > 0
       
      else: # Excluded by all 
        excl = excl == len(exclude_regions)
      
      keep = ~excl
      
      regions = regions[keep]
      centers = centers[keep]
      values = values[keep]
      
      if label_dict:
        labels = [x for i, x in enumerate(labels) if keep[i]]
    
    m0 = len(values)
    
    if m0:
      filt_region_dict[chromo] = regions
      filt_value_dict[chromo] = values
      filt_label_dict[chromo] = labels
      m += m0
      
    util.info('  .. chromosome/contig {} : keep {:,} of {:,} regions'.format(chromo, m0, n0), line_return=False)

  util.info('Keeping {:,} ({:.2f}%) of {:,} input regions'.format(m, (float(m*100.0)/n), n))
  
  bed.save_bed_data_track(out_bed_path, filt_region_dict, filt_value_dict, filt_label_dict)
  util.info('Written {}'.format(out_bed_path))
  
  return out_bed_path


def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(metavar='BED_FILE', dest='i',
                         help='Input data track file to be filtered, in BED format.')

  arg_parse.add_argument('-o', '--out-file', metavar='BED_FILE', dest='o',
                         help='Optional output BED file name containing filtered data regions. If not specified, a default based on the input filename will be used.')

  arg_parse.add_argument('-y', '--include-region-file', metavar='BED_FILES', nargs='*', dest='y',
                         help='Zero or more BED format files for which close or intersecting regions will be used to positively select input data track regions. '\
                              'If no files are specified all input regions will be selected prior to any exclusions.')

  arg_parse.add_argument('-n', '--exclude-region-file', metavar='BED_FILES', nargs='*', dest='n',
                         help='Zero or more BED format files for which close or intersecting regions will be used to exclude input data track regions.')
  
  arg_parse.add_argument('-sy', '--separation-max-include', metavar='SEQ_SEPARATIONS', nargs='*', type=int, dest="sy",
                         help='Maximum inclusion region (base pair); regions to be included must have centers within this seq. sparation of a filtering region. ' \
                              'May be set to a single value, which will be used for all inclusions. Otherwise a value can be specified separately '
                              'for the different filtering data files. Default: %.2f (for all region types)' % DEFAULT_SEP)
  
  arg_parse.add_argument('-sn', '--separation-max-exclude', metavar='SEQ_SEPARATIONS', nargs='*', type=int, dest="sn",
                         help='Maximum exclusion region (base pair); regions to be excluded must have centers within this seq. sparation of a filtering region. ' \
                              'May be set to a single value, which will be used for all exclusions. Otherwise a value can be specified separately '
                              'for the different filtering data files. Default: %.2f (for all region types)' % DEFAULT_SEP)
  
  arg_parse.add_argument('-y1', '--any-include', default=False, action='store_true', dest='y1', 
                         help='If set, any type of include region is sufficient for postively selecting an input region; an OR operation. ' \
                              'Otherwise the default is to select only where all types if include region are present; an AND operation')

  arg_parse.add_argument('-n1', '--exclude-any', default=False, action='store_true', dest='n1',
                         help='If set, any type of exclude region is sufficient for excluding an input region. Otherwise the default is to exclude where all types if exclude region are present')

  arg_parse.add_argument('-ql', '--quantile-min', default=DEFAULT_QUANTILE_MIN,
                         metavar='QUANTILE_VALUE_MIN', type=float, dest="ql",
                         help='The minimum quantile value to accept, as a percentile of all input values. Regions with values below this threshold will be excluded. ' \
                              'Default: %.2f' % DEFAULT_QUANTILE_MIN)

  arg_parse.add_argument('-qu', '--quantile-max', default=DEFAULT_QUANTILE_MAX,
                         metavar='QUANTILE_VALUE_MIN', type=float, dest="qu",
                         help='The maximum quantile value to accept, as a percentile of all input values. Regions with values above this threshold will be excluded. ' \
                              'Default: %.2f' % DEFAULT_QUANTILE_MAX)

  args = vars(arg_parse.parse_args(argv))
                                
  data_path = args['i']
  out_path = args['o']
  
  include_paths = args['y']
  exclude_paths = args['n']
  
  include_any = args['y1']
  exclude_any = args['n1']
  
  include_seps = args['sy']
  exclude_seps = args['sn']
  
  quantile_lower = args['ql']
  quantile_upper = args['qu']
  
  if not (include_paths or exclude_paths):
    if (quantile_lower == 0.0) and (quantile_upper == 100.0):
      util.critical('No include files (-y) or exclude files (-n) or quantile limits (-ql, -qu) specified')
    
  for in_path in [data_path] + include_paths + exclude_paths:
    io.check_invalid_file(in_path)
  
  if not (0.0 <= quantile_lower <= 100.0):
    util.critical('Lower quantile limit must be between 0.0 and 100.0')

  if not (0.0 <= quantile_upper <= 100.0):
    util.critical('Upper quantile limit must be between 0.0 and 100.0')
  
  if len(include_seps) not in (1, len(include_paths)):
    util.critical('The number of inclusion region separations (-sy) must match the number of inclusion region files (-y) or contain only a single value')
    
  if len(exclude_seps) not in (1, len(exclude_paths)):
    util.critical('The number of exclusion region separations (-sn) must match the number of inclusion region files (-n) or contain only a single value')
    
  data_track_filter(data_path, out_path, include_paths, exclude_paths, quantile_lower, quantile_upper, include_seps, exclude_seps, include_any, exclude_any)
  
if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()


"""
./nuc_tools data_track_filter /data/bed/Promoters.bed.gz -y /data/bed/H3K4me3_hap_EDL.bed -sy 500 -n /data/bed/H3K9me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed -sn 500 1000 -n1 -ql 25 -o avtive_promoters.bed
./nuc_tools data_track_filter /data/bed/H3K4me3_hap_EDL.bed -y /data/bed/Promoters.bed.gz -sy 500 -n /data/bed/H3K9me3_hap_EDL.bed /data/bed/H3K27me3_hap_EDL.bed -sn 500 1000 -n1 -ql 25 -o avtive_promoters.bed
"""



