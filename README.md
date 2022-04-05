# Robust Hierarchical Sparse Connectivity Patterns

MATLAB based tool for extracting hierarchical Sparse Connectivity Patterns (hSCPs) while reducing the effects of covariates such as age, sex and site in multi-site datasets. Standard hSCP implementation can be found at https://github.com/DushyantSahoo/hSCP. Implementation of robust hSCPs based on the following references:

- [1] Dushyant Sahoo and Christos Davatzikos. Learning Robust Hierarchical Patterns of Human Brain across Many fMRI Studies. NeurIPS 2021. https://ieeexplore.ieee.org/abstract/document/9285290
- [2] Dushyant Sahoo et al.Generalizable Hierarchical Patterns for identifying MDD patients: A Multisite Study. https://arxiv.org/pdf/2202.11144.pdf

contact: sahoodushyant@gmail.com

## Table of content
- [1. Overview](#id-section1)
- [2. Installation](#installation)
- [3. Main function](#main-function)

## Installation

Requirements:

* ``MATLAB``
* ``MATLAB Deep Learning Toolbox``

### Projection onto the intersection of the L1-ball and L-infinity-ball

Projection folder contains a Matlab/Mex/C++ implementation of a linear time algorithm for the projection onto the intersection of the L1-ball and L-infinity-ball (box constraint) provided by A. Podosinnikova.

### Using projection code with Matlab

- make sure your Matlab recognizes your gcc compiler
```
mex -setup C++
```
- compile the ```proj_L1_box.cpp``` function
```
install.m
```
- see an example of the code's usage
```
example_projection.m
```

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


## References

- [1] Sahoo, Dushyant, and Christos Davatzikos. "Learning Robust Hierarchical Patterns of Human Brain across Many fMRI Studies." Advances in Neural Information Processing Systems 34 (2021).
- [2] Sahoo, Dushyant, et al. "Robust Hierarchical Patterns for identifying MDD patients: A Multisite Study." arXiv preprint arXiv:2202.11144 (2022).
- [3] Podosinnikova, Anastasia. Robust Principal Component Analysis as a Nonlinear Eigenproblem. Diss. Universität des Saarlandes Saarbrücken, 2013.
- [4] Reddi, Sashank J., Satyen Kale, and Sanjiv Kumar. "On the convergence of adam and beyond." arXiv preprint arXiv:1904.09237 (2019).
