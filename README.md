"""Still editing, not exactly working."""

# ProMS: protein marker selection using proteomics or multi-omics data

ProMS (*Pro*tein *M*arker *S*election) is a python package designed to tackle a novel problem of multiview
feature selection: given multiple views of same set of samples,
select important features from one view of interest (target view) by integrating
information from other views. It was initially developed to select protein
biomarkers using proteomics data with the help of additional omics views
(e.g. RNAseq).


I edited ProMS to include three additional data integration strategies:
1, Late integration: The original ProMS algorithm, 
  which builds separate models for each omics view and combines the results based on selected features.

2, Mid integration: During model construction,
  both test and control groups must include at least one replicate from each omics view.

3, Early (pre) integration: Omics data are integrated prior to model construction.





original resources
[Check out the documentation](http://docs.zhang-lab.org/proms/)
