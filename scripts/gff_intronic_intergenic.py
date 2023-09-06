import sys, os, time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from collections import defaultdict
from nuc_tools import util, io
from formats import bed, gff


"""
Extract intergenic and intronic regions for a GFF files exon and gene regions.
"""


def get_gff_chr_regions(gff_file, min_gene_len=1e3):
  
  data_dict = gff.load_gene_dict(gff_path)
  
  intergenic_chr_regions = defaultdict(list)
  
  tiny_gene_chr_regions = defaultdict(list)
  
  short_gene_chr_regions = defaultdict(list) #
  short_intron_chr_regions = defaultdict(list) #
  short_first_exon_chr_regions = defaultdict(list) #
  short_midd_exon_chr_regions = defaultdict(list) #
  short_last_exon_chr_regions = defaultdict(list) #
   
  intron_chr_regions = defaultdict(list) #
  long_gene_chr_regions  = defaultdict(list) #
  first_exon_chr_regions = defaultdict(list) #
  midd_exon_chr_regions = defaultdict(list) #
  last_exon_chr_regions = defaultdict(list) #
  
  chromos = sorted(data_dict)
  
  for chromo in chromos:  
  
    gene_dict = data_dict[chromo]
    
    gene_regions = sorted([x[1:4] for x in gene_dict.values()])
    end_prev = -1
    
    for i, (gene_start, gene_end, gene_strand) in enumerate(gene_regions):
      gene_strand = 1 if gene_strand == '+' else 0      
      
      if i > 0:
        intergenic_chr_regions[chromo].append((end_prev, gene_end, gene_strand))
            
      end_prev = gene_end
    
    for gene in gene_dict:
      
      name, gene_start, gene_end, gene_strand, subfeats = gene_dict[gene]
      gene_strand = 1 if gene_strand == '+' else 0      
      
      gene_size = gene_end-gene_start
      
      if gene_size < 50:
        continue
      
      is_short_gene = gene_size < min_gene_len
      
      if is_short_gene:
        short_gene_chr_regions[chromo].append((gene_start, gene_end, gene_strand))
        
        if 100 < gene_size < 300:
          tiny_gene_chr_regions[chromo].append((gene_start, gene_end, gene_strand))
        
      else:
        long_gene_chr_regions[chromo].append((gene_start, gene_end, gene_strand))
      
      if 'exon' in subfeats:
        exons = sorted(subfeats['exon'])
        last = len(exons)-1
        end_prev = -1
        
        for i, (start, end, strand) in enumerate(exons):
          
          exon_size = end - start
          
          if i == 0:
            if is_short_gene:
              short_first_exon_chr_regions[chromo].append((start, end, gene_strand))
            else:
              first_exon_chr_regions[chromo].append((start, end, gene_strand))
          
          elif i == last:
            if is_short_gene:
              short_last_exon_chr_regions[chromo].append((start, end, gene_strand))
            else:
              last_exon_chr_regions[chromo].append((start, end, gene_strand))
          
          else:
            if is_short_gene:
              short_midd_exon_chr_regions[chromo].append((start, end, gene_strand))
            else:
              midd_exon_chr_regions[chromo].append((start, end, gene_strand))
          
          if i > 0:
            if is_short_gene:
              short_intron_chr_regions[chromo].append((end_prev, start, gene_strand))
            else:
              intron_chr_regions[chromo].append((end_prev, start, gene_strand))
          
          end_prev = end

    tiny_gene_chr_regions[chromo] = np.array(tiny_gene_chr_regions[chromo], int)
    
    intergenic_chr_regions[chromo] = np.array(intergenic_chr_regions[chromo], int)

    short_gene_chr_regions[chromo] = np.array(short_gene_chr_regions[chromo], int)
    short_intron_chr_regions[chromo] = np.array(short_intron_chr_regions[chromo], int)
    short_first_exon_chr_regions[chromo] = np.array(short_first_exon_chr_regions[chromo], int)
    short_midd_exon_chr_regions[chromo] = np.array(short_midd_exon_chr_regions[chromo], int)
    short_last_exon_chr_regions[chromo] = np.array(short_last_exon_chr_regions[chromo], int)
    
    long_gene_chr_regions[chromo] = np.array(long_gene_chr_regions[chromo], int)
    intron_chr_regions[chromo] = np.array(intron_chr_regions[chromo], int)
    first_exon_chr_regions[chromo] = np.array(first_exon_chr_regions[chromo], int)
    midd_exon_chr_regions[chromo] = np.array(midd_exon_chr_regions[chromo], int)
    last_exon_chr_regions[chromo] = np.array(last_exon_chr_regions[chromo], int)
  
  chr_region_dicts = [intron_chr_regions, first_exon_chr_regions,
                      midd_exon_chr_regions, last_exon_chr_regions,
                      intergenic_chr_regions, long_gene_chr_regions,
                      short_gene_chr_regions, short_intron_chr_regions,
                      short_first_exon_chr_regions,  short_midd_exon_chr_regions,
                      short_last_exon_chr_regions, tiny_gene_chr_regions]
  
  return chr_region_dicts

def write_bed_regions(bed_file_path, chr_region_dict):

  data_dict = {}
  for chromo in chr_region_dict:
    regions = chr_region_dict[chromo]
    
    if len(regions):
      value_anno = np.ones(regions.shape)
      value_anno[:,2] = 0
 
      data_dict[chromo] = np.concatenate([regions, value_anno], axis=1)

  bed.save_data_track(bed_file_path, data_dict)


def gff_to_further_bed_regions(out_file_root, gff_file, min_gene_len=1000):
  
  chr_region_dicts = get_gff_chr_regions(gff_file, min_gene_len)
  
  tags = ('_introns.bed','_first-exon.bed',
          '_mid-exon.bed','_last-exon.bed',
          '_intergenic.bed','_long-gene.bed',
          '_short-gene.bed','_short-intron.bed',
          '_short-first-exon.bed','_short-mid-exon.bed',
          '_short-last-exon.bed','_tiny-gene.bed')
  
  for chr_region_dict, tag in zip(chr_region_dicts, tags):
    write_bed_regions(out_file_root + tag, chr_region_dict)
  

if __name__ == '__main__':
  
   #gff_path = '/data/dino_hi-c/hem_flye_4_ssRNA_Trinity.gff3'
   gff_path = '/data/dino_hi-c/hem_flye_4_unique_cds_v2.gff3'

   gff_to_further_bed_regions('/data/dino_hi-c/HEM_FLYE_4_unique_cds_v2', gff_path)
  
