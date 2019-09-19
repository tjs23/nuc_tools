# Nucleome processing and analysis toolkit

The nuc_tools software is a collection of nucleome analysis tools with an emphasis on genome structure.
Individual tools are provided for the processing, analysis and visualisation of chromatin contact data 
in combination with asscociated data like ChIP-seq.
This includes canonical, multi-cell Hi-C data as well as single-cell Hi-C. Some tools are also provided
for the analysis of whole genome structures that derive from single-cell Hi-C data.

The nuc_tools collection is a work-in-progress and will be expanded as time allows. 


## Python Module Dependencies

This software is designed to work with Python version 2.7 or 3+ and uses [scipy](https://www.scipy.org/),
[numpy](https://numpy.org/) and [matplotlib](https://matplotlib.org/) modules extensively. These dependencies
are readily installed via the SciPy stack (which includes NumPy and Mmatplotlib). This maye be installed using `pip`:

`sudo pip install scipy`

Alternatively, SciPy is contained in bundled Python packages like Anaconda or Canopy,
and may be installed via most Linux distributions' package managers.


## Other Dependencies

Some sub-tools have additional dependencies, which should be made available as command line executables,
i.e. adjusting $PATH if needs be.

On recent Ubuntu/Min/Debian Linux distributions these may be installed using:

`sudo apt-get install bowtie2 macs2`


## Running nuc_tools

After cloining the GitHub repository, nuc_tools can be invoked as an executable, specifying the specific tool name
as an argument:

 `./nuc_tools {tool_name} {tool_options}`

The currently available tools may be listed using

 `./nuc_tools -h`
 
And the command-line options for each tool is available using '-h' after the tool name:

 `./nuc_tools {tool_name} -h`
 

## Current tools

### chip-seq_process

An automated ChIP-seq processing pipline using Bowtie2 mapping and MACS2 peak calling.

### contact_compare

Differential Hi-C contact map analysis. Graphically presents the most significant differences in Hi-C contact
counts or compartmental correalation (e.g. A/B compartment changes). Outputs files in PDF format
and includes optional diagonal-only view.

### contact_insulation

Calculation and analysis of Hi-C contact insulation, for multiple datasets, at user-specified chromosomal potions.

### contact_map

Displays Hi-C contact maps, both bulk and single-cell, with superimposed genome data tracks. Outputs files in PDF format
and includes optional diagonal-only and dual sample (saplit diagonal) views.

### contact_pair_points

Analysis of Hi-C contact counts, for multiple datasets, at user-specified interation poistions (e.g. loop points).

### contact_probability

Displays the sequence separation verses Hi-C contact probability for one or more datasets.

### data_track_compare

A suite of pairwise genome data track analyses, output as PDF format.
Investigates co-localisation in genome sequence positions and data value correlations. 

### data_track_filter

Filters one genome data track according to the presence or absence of data in other different data tracks,
e.g. to look at the overlap of histone marks etc. Currently only works with BED format.

### ncc_bin

Convert the detailed NCC chromatin contact format text files, as output by the Hi-C processsing software (nuc_process), into
a highly compact, binned binary format. Uses a NumPy zipped archive (.npz) of sparse matrix representations with adaptive
integer sizes. 

### ncc_filter

Create subsets of NCC format chromatin contact data according to sequence separation, cis/trans assignment, chromosome,
mapping ambiguity etc.

### structure_compare

A suite of whole-genome structure analyses, output as PDF format.
Performs RMSD (coordinate precision) analysis of alternative models within a single structure and between different
structures. Includes both global and (chromosomal) postion-specific measures. 

### structure_data_density

Analyses the co-localisation of genome data tracks on 3D, single-cell genome structures ising a density based appraoch.
Includes both the self-self co-localisation of individual data tracks and the co-localisation of one data track with another.
A variety of different random/null hypotheses are provided to investigate global or local co-clustering trends.

