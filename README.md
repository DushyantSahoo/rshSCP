# Robust Hierarchical Sparse Connectivity Patterns

MATLAB based tool for extracting hierarchical Sparse Connectivity Patterns (hSCPs) while reducing the effects of covariates such as age, sex and site in multi-site datasets. Implementation of robust principal component analysis and stable principal component pursuit based on the following references:

contact: sahoodushyant (at) gmail (dot) com

## Table of content
- [1. Overview](#id-section1)
- [2. Installation](#id-section2)
- [3. Quick Start](#id-section3)

**References**: If you are using ComBat for the harmonization of multi-site imaging data, please cite the following papers:


## Installation

If you are not using a distribution like TeX Live or MikTeX, you can
easily install the package by running (on a command line; the $ signals
denotes the prompt and should not be typed):

    $ tex algorithms.dtx

This should generate, among others, the files `algorithm.sty` and
`algorithmic.sty`. To use them, just copy them to your texmf tree (or
the local directory where the document you want to typeset resides).  If
you would like to generate the documentation, just use, say:

    $ pdflatex algorithms.dtx
    $ pdflatex algorithms.dtx


## Development and Support

The author welcomes any contribution and also tries to address any bugs
or feature requests that may be filed on the issue tracker at
<https://github.com/DushyantSahoo/rshSCP/issues>.


Citations
---------

Here's a list of other related projects where you can find inspiration for
creating the best possible README for your own project:

- [Hierarchical extraction of functional connectivity components in human brain using resting-state fMRI](https://ieeexplore.ieee.org/abstract/document/9285290)
- [Extraction of Hierarchical Functional Connectivity Components in human brain using Adversarial Learning](https://arxiv.org/pdf/2104.10255.pdf)
- [Learning Robust Hierarchical Patterns of Human Brain across Many fMRI Studies](https://proceedings.neurips.cc/paper/2021/file/f33ba15effa5c10e873bf3842afb46a6-Paper.pdf)
- [Robust Hierarchical Patterns for identifying MDD
patients: A Multisite Study](https://arxiv.org/pdf/2202.11144.pdf)

.. [1] Pomponio, R., Shou, H., Davatzikos, C., et al., (2019).
   "Harmonization of large MRI datasets for the analysis of brain imaging
   patterns throughout the lifespan." Neuroimage 208.
   https://doi.org/10.1016/j.neuroimage.2019.116450.
.. [2] Fortin, J. P., N. Cullen, Y. I. Sheline, W. D. Taylor, I. Aselcioglu,
   P. A. Cook, P. Adams, C. Cooper, M. Fava, P. J. McGrath, M. McInnis,
   M. L. Phillips, M. H. Trivedi, M. M. Weissman and R. T. Shinohara (2017).
   "Harmonization of cortical thickness measurements across scanners and sites."
   Neuroimage 167: 104-120. https://doi.org/10.1016/j.neuroimage.2017.11.024.
.. [3] W. Evan Johnson and Cheng Li, Adjusting batch effects in microarray
   expression data using empirical Bayes methods. Biostatistics, 8(1):118-127,
   2007. https://doi.org/10.1093/biostatistics/kxj037.
