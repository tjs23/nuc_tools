import os, sys
import numpy as np

from os.path import dirname

START = 's'
MIDDLE = 'm'
END = 'e'
INTERPOLATION_POS = (START,MIDDLE,END)

PROG_NAME = 'structure_data_coords'
VERSION = '1.0.0'
DESCRIPTION = 'Extract interpolated 3D coordinate positions for data track values (e.g. peaks)'
DEFAULT_FN_PAT = 'Data_3D_coords_%s_%s.tsv'
HEADERS = ('chromosome', 'start', 'end', 'label', 'strand', 'value', 'x_mean', 'y_mean', 'z_mean', 'rmsd')

def structure_data_coords(struc_paths, data_tracks, struc_labels=None,
                          data_labels=None, out_dir='./',
                          out_file_patt=DEFAULT_FN_PAT, interp_pos=MIDDLE,
                          percentile_threshold=None):
  
  from nuc_tools import io, util
  from formats import bed, n3d
  
  msg = 'Analysing {} data tracks with {} structures.'
  util.info(msg.format(len(data_tracks), len(struc_paths)))
  
  data_labels = io.check_file_labels(data_labels, data_tracks)

  struc_labels = io.check_file_labels(struc_labels, struc_paths)
  
  # chromosome, start, end, label, strand, value, x_mean, y_mean, z_mean, rmsd
  line_fmt = '%s\t%d\t%d\t%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%.5f\n'
  
  for s, n3d_path in enumerate(struc_paths):
    seq_pos_dict, coords_dict = n3d.load_n3d_coords(n3d_path)
    info_data = (os.path.basename(n3d_path), sum([len(coords_dict[c]) for c in coords_dict]), len(seq_pos_dict))
    util.info('Analysing structure {} : {} coordinates over {} chromosomes'.format(*info_data))
    
    coords_dict, model_rmsds, model_mean_rmsds, particle_rmsds = util.align_chromo_coords(coords_dict, seq_pos_dict, n_iter=1, dist_scale=True)
    
    struc_chromos = set(seq_pos_dict) 
    
    for d, data_path in enumerate(data_tracks):
      data_regions, data_values, d_labels = bed.load_bed_data_track(data_path)
      util.info(' .. with {} : {:,} values'.format(os.path.basename(data_path), sum([len(data_values[c]) for c in data_values])))
      data_chromos = set(data_values)
      
      if percentile_threshold:
        threshold = np.percentile(np.concatenate(data_values.values()), percentile_threshold)
      else:
        threshold = None
      
      chromos = struc_chromos & data_chromos
      
      if not chromos:
        util.critical('No chromosome labels in structure file {} match data file {}'.format(n3d_path, data_path))
      
      if len(chromos) < len(struc_chromos):
        missing = ' '.join(sorted(struc_chromos-data_chromos))
        util.warn(' !! Not all structure chromosomes represented in data file. Missing: %s' % missing)
      
      if len(chromos) < len(data_chromos):
        missing = ' '.join(sorted(data_chromos-struc_chromos))
        util.warn(' !! Not all data file chromosomes represented in structure: Missing: %s' % missing)
      
      out_file_name = out_file_patt % (struc_labels[s].replace(' ','_'), data_labels[d].replace(' ','_'))
      out_file_path = os.path.join(out_dir, out_file_name)
      
      chromos = util.sort_chromosomes(chromos)
      
      with open(out_file_path, 'w') as out_file_obj:
        write = out_file_obj.write
        write('#' + '\t'.join(HEADERS) + '\n')
         
        for chromo in chromos:
          seq_pos = seq_pos_dict[chromo]
          coords = coords_dict[chromo]
          values = data_values[chromo]
          regions = data_regions[chromo]
          labels = d_labels[chromo]
          bin_size = float(seq_pos[1]-seq_pos[0])
 
          n_part = len(seq_pos)
          is_stranded = min(np.diff(regions, axis=1)) < 0
          
          if interp_pos == START:
            pos = regions[:,0]
          elif interp_pos == END:
            pos = regions[:,1]
          else:
            pos = regions.mean(axis=1)
 
          if threshold is not None:
            idx = (values > threshold).nonzero()
            pos = pos[idx]
            values = values[idx]
            regions = regions[idx]
            labels = [labels[i] for i in idx]
            
          if not len(values):
            continue
 
          pidx = np.searchsorted(seq_pos, pos)

          valid = (pidx >= 0) & (pidx < n_part-1)
          pidx = pidx[valid] # Data could be outside coord bounds
          pos = pos[valid]
          values = values[valid]
          regions = regions[valid]
          labels = [labels[i] for i, x in enumerate(valid) if x]
          
          
          fracs = (pos - seq_pos[pidx]).astype(float)/bin_size # How close to the next particle
          fracs = fracs[:,None]
          
          data_coords = fracs * coords[:,pidx+1] + (1.0 - fracs) * coords[:,pidx]
          mean_coords = data_coords.mean(axis=0)
          model_mean_rmsds, point_rmsds = util.calc_rmsds(mean_coords, data_coords)
          
          for i, value in enumerate(values): # New, orig indices
            start, end = regions[i]
            label = labels[i]
            
            if is_stranded:
              if start > end:
                strand = '-'
                end, start = start, end
              else:
                strand = '+'
            else:
              strand = '.'
              
            x_mean, y_mean, z_mean = mean_coords[i]
            rmsd = point_rmsds[i]
            
            line = line_fmt % (chromo, start, end, label, strand, value, x_mean, y_mean, z_mean, rmsd)
            write(line)

      util.info(' .. Wrote %s' % out_file_name)

def main(argv=None):

  from argparse import ArgumentParser
  from nuc_tools import util, io
  
  if argv is None:
    argv = sys.argv[1:]

  epilog = 'For further help email tjs23@cam.ac.uk or wb104@cam.ac.uk'

  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)
  
  arg_parse.add_argument(metavar='N3D_FILES', nargs='+', dest='i',
                         help='One or more genome structure files in N3D format. Accepts wildcards.')

  arg_parse.add_argument('-d', '--data-tracks', metavar='BED_FILES', nargs='+', dest='d',
                         help='One or more data track files in BED format. Accepts wildcards.')

  arg_parse.add_argument('-s', '--structure-labels', metavar='STRUCTURE_NAME', nargs='+', dest='s',
                         help='Optional textual labels/names for each input structure.')

  arg_parse.add_argument('-l', '--data-labels', metavar='DATA_NAME', nargs='+', dest='l',
                         help='Optional textual labels/names for each input data tracks.')
 
  arg_parse.add_argument('-o', '--out-dir', metavar='OUT_DIR', default='./', dest='o',
                         help='Optional directory for saving output data. Defaults to the current working directory.')
  
  arg_parse.add_argument('-p', '--position', default=MIDDLE, metavar='S/M/E', dest="p",
                         help='Which position of a data region (e.g. peak) to use for 3D interpolation. Must be ' \
                              '"%s" for start, "%s" for middle or "%s" for end. Default: %s' % (START, MIDDLE, END, MIDDLE))

  arg_parse.add_argument('-m', '--percentile-min', default=None,
                         metavar='PERCENTILE', type=float, dest="m",
                         help='Optional minimum percentile, only above which data values will be used. ' \
                              'E.g. 75 uses the top quartile of data.')  

  arg_parse.add_argument('-f', '--filename-fmt', metavar='FILENAME_FMT', default=DEFAULT_FN_PAT, dest='f',
                         help='Optional formatting template for creating file names from combinations ' \
                              'of strucure and data label. Default: %s' % DEFAULT_FN_PAT.replace('%','%%'))
           
  args = vars(arg_parse.parse_args(argv))
                                
  struc_paths = args['i']
  data_tracks = args['d']
  struc_labels = args['s'] or None
  data_labels = args['l'] or None
  
  out_dir = args['o'] or './'
  interp_pos = args['p'].lower()
  pct_threshold = args['m']
  out_file_patt = args['f']
    
  if not struc_paths:
    arg_parse.print_help()
    sys.exit(1)  
  
  for in_path in struc_paths + data_tracks:
    io.check_invalid_file(in_path)
  
  if pct_threshold is not None:
    if not (0.0 < pct_threshold < 100.0):
      util.critical('Percentile threshold must be greater than zero and less than 100')
  
  if interp_pos not in INTERPOLATION_POS:
    util.critical('Interpolation position must be "%s", "%s" or "%s"' % (START, MIDDLE, END))
    
  try:
    test = out_file_patt % ('a','b')
  except TypeError:
    util.critical('Unusable filename template "%s"' % (out_file_patt))

  structure_data_coords(struc_paths, data_tracks, struc_labels, data_labels, out_dir,
                        out_file_patt, interp_pos, pct_threshold)
                          
if __name__ == "__main__":
  sys.path.append(dirname(dirname(__file__)))
  main()                          


"""
./nuc_tools structure_data_coords  P35E5.n3d -d TF_BS_list_all_2.bed
"""
                          
