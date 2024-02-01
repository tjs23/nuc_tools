import sys, os, time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict
from nuc_tools import util, io
from formats import bed, gff


"""
Extract intergenic and intronic regions for a GFF files exon and gene regions.
"""

REGION_KEYS = ['intergenic','tiny_gene','short_gene',
               'short_intron','short_first_exon',
               'short_midd_exon','short_last_exon',
               'intron','long_gene','first_exon',
               'midd_exon','last_exon']

def get_gff_chr_regions(gff_file, gene_min=10, tiny_gene_max=300, small_gene_max=1000):
  
  data_dict = gff.load_gene_dict(gff_path)
  
  chromo_region_dicts = {}
  for key in REGION_KEYS:
    chromo_region_dicts[key] =  defaultdict(list)
  
  chromos = sorted(data_dict)
  
  for chromo in chromos:  
    gene_dict = data_dict[chromo]
    gene_regions = sorted([x[1:4] for x in gene_dict.values()])
    end_prev = -1
    
    for i, (gene_start, gene_end, gene_strand) in enumerate(gene_regions):
      gene_strand = 1 if gene_strand == '+' else 0      
      
      if i > 0:
        chromo_region_dicts['intergenic'][chromo].append((end_prev, gene_end, gene_strand))
            
      end_prev = gene_end
    
    for gene in gene_dict:
      name, gene_start, gene_end, gene_strand, subfeats = gene_dict[gene]
      gene_strand = 1 if gene_strand == '+' else 0      
      gene_size = gene_end-gene_start
      
      if gene_size < gene_min:
        continue
      
      is_short_gene = gene_size < small_gene_max
      
      if is_short_gene:
        gene_type = 'tiny_gene' if gene_size < tiny_gene_max else 'short_gene'
      
      else:
        gene_type = 'long_gene'
      
      chromo_region_dicts[gene_type][chromo].append((gene_start, gene_end, gene_strand))
      
      if 'exon' in subfeats:
        exons = sorted(subfeats['exon'])
        last = len(exons)-1
        end_prev = -1
        
        for i, (start, end, strand) in enumerate(exons):
          if i == 0:
            exon_type = 'short_first_exon' if is_short_gene else 'first_exon'
          elif i == last:
            exon_type = 'short_last_exon' if is_short_gene else 'last_exon'
          else:
            exon_type = 'short_midd_exon' if is_short_gene else 'midd_exon'
  
          chromo_region_dicts[exon_type][chromo].append((start, end, gene_strand))
           
          if i > 0:
            intron_type = 'short_intron' if is_short_gene else 'intron'
            chromo_region_dicts[intron_type][chromo].append((end_prev, start, gene_strand))
          
          end_prev = end
   
    for key in chromo_region_dicts:
      chromo_region_dicts[key][chromo] =  np.array(chromo_region_dicts[key][chromo], int)
    
  return chromo_region_dicts


def write_bed_regions(bed_file_path, chr_region_dict):

  data_dict = {}
  for chromo in chr_region_dict:
    regions = chr_region_dict[chromo]
    
    if len(regions):
      value_anno = np.ones(regions.shape)
      value_anno[:,2] = 0
 
      data_dict[chromo] = np.concatenate([regions, value_anno], axis=1)

  bed.save_data_track(bed_file_path, data_dict)


def gff_to_gene_regions(out_file_root, gff_file, gene_min=10, tiny_gene_max=300, small_gene_max=1000):
  
  print(f'Reading {gff_file}')
  chr_region_dicts = get_gff_chr_regions(gff_file, gene_min, tiny_gene_max, small_gene_max)
  
  for key in chr_region_dicts:
    file_path = f'{out_file_root}_{key}.bed'
    n = sum([len(x) for x in chr_region_dicts[key].values()])
    write_bed_regions(file_path, chr_region_dicts[key])
    print(f' - Wrote {n:,} "{key}" regions to {file_path}')

if __name__ == '__main__':
   
   # Basic Trinity transcriptome assembly
   #gff_path = '/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3'
   
   # Refined transcriptome assambly with unique CDS
   gff_path = '/data/dino_hi-c/hem_flye_4_unique_cds_v2.gff3'
   
   out_file_root = '/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2'

   gff_to_gene_regions('/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2', gff_path)
  
