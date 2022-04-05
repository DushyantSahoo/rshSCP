# Robust Hierarchical Sparse Connectivity Patterns

## Table of content
- [1. Introduction](#id-section1)
- [2. Software implementations](#id-section2)
- [3. Missing values](#id-section3)
- [4. FAQs](#id-section3)
- [5. News](#id-section3)

**References**: If you are using ComBat for the harmonization of multi-site imaging data, please cite the following papers:



The `algorithms` bundle provides two environments, `algorithmic` and
`algorithm`, designed to typeset pseudo-code with LaTeX.

The `algorithmic` package provides an environment for the description of
algorithms, step-by-step in pseudo-code, while the `algorithm`
environment provides a `float` wrapper for algorithms to "feature" them
in the text.

Even though the primary intention is to use them together, it is
possible to just use one of them at the option of the user.  The reason
for two environments being provided is to allow the user maximum
flexibility.


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

The source code for the bundle is currently hosted in a `git` repository at
<https://github.com/rbrito/algorithms>.

The author welcomes any contribution and also tries to address any bugs
or feature requests that may be filed on the issue tracker at
<https://github.com/rbrito/algorithms/issues>.


## The License

The algorithms bundle is (currently) licensed under the Lesser GPL
license, which is a Free Software license. It may, in the future, be
released under the standard LaTeX license (the LaTeX Project Public
License).


 -- Rogério Brito <rbrito@ime.usp.br>  Wed, 08 Nov 2017 05:23:56 -0200
