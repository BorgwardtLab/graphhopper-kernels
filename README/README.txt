README

v1.3 (14.1.2016)  Updated GraphHopper_dataset.m to avoid double calculations and numerical imprecision resulting in unsymmetric kernel matrices. Fixed typo in example.m. Removed dependencies on MatlabBGL, which does not work with newer Matlab versions. (Thank you to Marco Alvarez who spotted the issues!) Removed use of matlabpool, which is no longer part of Matlab. 
v1.2 (8.4.2013)   Data corrected and updated; error corrected in propagation_kernel.m.
v1.1 (31.10.2013) Original version

This is a matlab implementation of the GraphHopper kernel first presented in the paper 

[1] 'Scalable kernels for graphs with continuous attributes' by A. Feragen, N. Kasenburg, J. Petersen, M. de Bruijne a d K. Borgwardt, published at Neural Information Processing Systems (NIPS) 2013.

We also include our impementation of the propagation kernel presented in 

[2] 'Efficient graph kernels by randomization' by M. Neumann, N. Patricia, R. Garnett, and K. Kersting, ECML/PKDD (1), pages 378–393, 2012 (see also comment below).


CONTENTS

Matlab functions:

example.m (shows example usage of the implemented kernels)
GraphHopper_dataset.m (computes the GraphHopper kernel)
propagation_kernel.m and WL_propagation_kernel.m (computes the propagation kernel with diffusion and WL update rules, respectively)
normalize_kernel.m (Normalizing the Gram matrix)
d_vectors_dag.m
o_vectors_dag.m
runntimes.m (runs N 10-fold cross-validation SVM classification experiments)
runIndependent.m (runs a 10-fold cross-validation SVM classification experiment)


INSTALLATION

* Older versions (up to 1.2) of the function GraphHopper_dataset.m requires having the MatlabBGL toolbox by David Gleich installed. This can be downloaded from: www.mathworks.com/matlabcentral/fileexchange/10922-matlabbgl
* In older versions (up to 1.2), the function GraphHopper_dataset.m can take advantage of the Matlab Parallel Computing toolbox (www.mathworks.com/products/parallel-computing/). In order to allow your matlab to run parts of the function in parallel, execute
	'matlabpool local N'
prior to running GraphHopper_dataset, where N is the number of cores you would like to use. A version allowing use of Matlab's new parallel computing tools will be available soon.
* The functions runntimes.m and runIndependent.m assume that LibSVM along with its matlab interface is installed. This can be downloaded from: http://www.csie.ntu.edu.tw/~cjlin/libsvm/.


DATA

We are also releasing the following datasets used in the paper: ENZYMES, PROTEINS, SYNTHETIC. Note that the datasets ENZYMES and PROTEINS have previously appeared with discretized node features, this version of the data has continuous-valued features as described in our NIPS submission.

v1.2: The datasets ENZYMES_symmetrized and SYNTHETICnew are also added; see ERRATUM for details.


THE PROP KERNEL

Our implementation of the PROP kernel is different from those used in [2], as
the continuous label distributions described in [2] were replaced by continuous-valued features as
indicated in [2, page 4, footnote 3]. In personal communication, Marion Neumann has explained that propagating continuous-valued
features, as we do, leads to a smoothing, or averaging, of the continuous node features, leading to
poor ability to capture graph structure. This affects our implementation of PROP-diff. See the enclosed commentary for details.


CONTACT

Any questions can be directed to Aasa Feragen: aasa.feragen@tuebingen.mpg.de or aasa@diku.dk.
