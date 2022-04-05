# Robust Hierarchical Sparse Connectivity Patterns

MATLAB based tool for extracting hierarchical Sparse Connectivity Patterns (hSCPs) while reducing the effects of covariates such as age, sex and site in multi-site datasets. Implementation of robust hSCPs based on the following references:

- [1] Dushyant Sahoo and Christos Davatzikos. Learning Robust Hierarchical Patterns of Human Brain across Many fMRI Studies. NeurIPS 2021. https://ieeexplore.ieee.org/abstract/document/9285290
- [2] Dushyant Sahoo and Christos Davatzikos. Extraction of hierarchical functional connectivity components in human brain using adversarial learning. https://arxiv.org/pdf/2104.10255.pdf
- [3] Dushyant Sahoo et al.Generalizable Hierarchical Patterns for identifying MDD patients: A Multisite Study. (Under Review)
- [4] Dushyant Sahoo, T. D. Satterthwaite, and Christos Davatzikos. Hierarchical extraction of functional connectivity components in human brain using resting-state fmri. TMI 2020. https://ieeexplore.ieee.org/abstract/document/9285290

contact: sahoodushyant (at) gmail (dot) com

## Table of content
- [1. Overview](#id-section1)
- [2. Installation](#Installation)
- [3. Quick Start](#main-function)

## Installation

Requirements:

* ``MATLAB``
* ``MATLAB Deep Learning Toolbox``

## Main function

The current implementation requires a positive semi-definite symmetric matrix, for example, a correlation matrix as an input. Below is the main function, and description about input and output. 

```[W, lambda, error] = hscp_amsgrad(A,k,alpha,loop,eta,beta1,beta2,eps,tole,svd_check)```

Below are the inputs to the above function

1) A contains the input matrices in in a cell format, for example- A{1}, A{2}, ....

2) k contains the number of components in each hierarchy, for example k = [120,20,4] where 120 is the number of nodes in data or the size of input matrix
 20 is the number of components at level 1,
 4 is the number of components at level 2.
 k can also be [120,10] having only one level or [120,40,20,4] having
 three levels

3) alpha is the sparsity level of each components at each level, you will
 have to play with it a little bit to get a balance between sprasity in
 the components and good approximation error.
 If k = [120,20,4] then alpha can be [1,1] i.e. sparsity for each level

 4) loop is the number of iterations of gradient descent

 5) eta, beta1, beta2, eps are the hyperparameters for amsgrad

 6) tole is the % change in the error before gradient descent stops

 7) svd_check is used for initializing the algorithm with SVD, details are
 given in the main paper
 
 Below are the typical hyperparameter settings that would work-
 1) svd_check = 1
 2) loop = 6000
 3) eta = 0.1
 4) beta1 = 0.99
 5) beta2 = 0.999
 6) eps = 10^-8
 7) tol1 = 10^(-4)

 There are three outputs-
 1) W stores components at different level in cells, each cell of W will
 store components at each level
 2) lambda stores subject specific information in cell, each cell of
 lambda will store subject and level specific information
 3) error stores % information captured by the decomposition, ideally it
 should be decreasing with the iterations.

MATLAB for simulating data and running the code is also given which would give user an idea of the input parameters and how the output looks. Please refer to "Hierarchical extraction of functional connectivity components in human brain using resting-state fMRI" paper for more details. I am thankful to Anastasia for providing me with the code for projection operators.

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
