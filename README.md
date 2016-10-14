# Scalable kernels for graphs with continuous attributes

## Code

A MATLAB implementation of the GraphHopper kernel as proposed in [1]. An implementation of the propagation kernel as presented in [2] is also included. Installation instructions are below.

[1] A. Feragen, N. Kasenburg, J. Petersen, M. de Bruijne a d K. Borgwardt (2013) **Scalable kernels for graphs with continuous attributes**, _Advances in Neural Information Processing Systems 26 (NIPS 2013)_ [link](https://papers.nips.cc/paper/5155-scalable-kernels-for-graphs-with-continuous-attributes)  

[2] M. Neumann, N. Patricia, R. Garnett, and K. Kersting (2012) **Efficient graph kernels by randomization**, _ECML/PKDD_ (0), 377–392 [link](http://link.springer.com/chapter/10.1007%2F978-3-642-33460-3_30)

## Installation

The functions `runntimes.m` and `runIndependent.m` assume that LibSVM along with its MATLAB interface is installed. This can be downloaded from: [http://www.csie.ntu.edu.tw/~cjlin/libsvm/.](http://www.csie.ntu.edu.tw/~cjlin/libsvm/.)


## Datasets

The following datasets used in the paper are available in the folder `CodeandData`: 
* ENZYMES 
* PROTEINS 
* SYNTHETIC

as well as 

* ENZYMES\_symmetrized 
* SYNTHETICnew  

Note that the datasets ENZYMES and PROTEINS have previously appeared with discretized node features, this version of the data has continuous-valued features as described in our NIPS submission.

Please, refer to the file `README/README.txt` for further information on usage and installation of previous versions.


[//]: # (## Author)

[//]: # (Aasa Feragen)


## Contact 

Any questions can be directed to Aasa Feragen: aasa [at] di.ku.dk
