# ProMS3: protein marker selection using proteomics or multi-omics data with 3 algorithms

ProMS (*Pro*tein *M*arker *S*election) is a python package designed to tackle a novel problem of multiview
feature selection: given multiple views of same set of samples,
select important features from one view of interest (target view) by integrating
information from other views. It was initially developed to select protein
biomarkers using proteomics data with the help of additional omics views
(e.g. RNAseq).


I edited ProMS to switch three data integration strategies:

1, Late integration (post): The original ProMS algorithm, 
  which builds separate models for each omics view and combines the results based on selected features.

2, Mid integration (mid): During model construction,
  both test and control groups must include at least one replicate from each omics view.

3, Early (pre) integration: Omics data are integrated prior to model construction.


branch origin resources
[Check out the documentation](http://docs.zhang-lab.org/proms/)



## 💻 Installation and Run ProMS3
</code></pre>
```bash
# We recommend building a new conda environment
conda create -n proms3 python=3.11
conda activate proms3

# Install dependencies
conda install -c conda-forge ecos

# Install proms3 directly from GitHub
pip install git+https://github.com/ta-akb/proms3

# Confirm installation
which proms3
# This should show the path to the installed CLI if successful
# ex, /opt/homebrew/Caskroom/miniforge/base/envs/proms3/bin/proms3

# Download the test_files directory and navigate into it

# Example execution
proms3_train -f crc_run_conf.yml -d crc_data_conf3_pre.yml
```
</code></pre>

## 📊 Example Output

Here is an example of the test file result on terminal with results directory creation:

![UMAP result](docs/images/results.png)


## 📄 Reference

This package is part of an ongoing research project.  
The associated manuscript is currently being prepared for submission.

