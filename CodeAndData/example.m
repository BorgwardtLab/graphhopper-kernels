% This is a script showing example usage of the GraphHopper and propagation
% kernel implementations.

%--------------------------------------------------------------------------

% 1. Load the ENZYMES dataset, and for the sake of illustration, pick out
% 10 graphs from each of two classes

load('ENZYMES.mat')
enz = ENZYMES(1:10);
enz(11:20) = ENZYMES(end-9:end);
lenz =[lenzymes(1:10), lenzymes(end-9:end)];

%--------------------------------------------------------------------------

% 2. Allocate 4 cores for parallel computation

% pool = parpool(4);

%--------------------------------------------------------------------------

% 3. Compute the GraphHopper kernel

[K, comp_time] = GraphHopper_dataset(enz, 'gaussian', 1/3, 1);

%--------------------------------------------------------------------------

% 4. Close the matlabpoool

% matlabpool close force

%--------------------------------------------------------------------------

% 5. Normalize the kernel

K_n = normalize_kernel(K);

%--------------------------------------------------------------------------

% 6. Run a 10-fold cross-validation classification experiment with libSVM

result = runntimes(K_n,lenz, 10);

%--------------------------------------------------------------------------

% 7. Compute the propagation kernel, normalize it and run another
% classification experiment

[K, runtime] = propagation_kernel(enz,10,0.00001,1,0);
K_n = normalize_kernel(K{end});
result = runntimes(K_n,lenz, 10);

%--------------------------------------------------------------------------

% 8. Compute the WL_propagation kernel, normalize it and run another
% classification experiment

[K, runtime] = WL_propagation_kernel(enz,10,0.00001,1,0);
K_n = normalize_kernel(K{end});
result = runntimes(K_n,lenz, 10);
