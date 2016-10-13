% propagation_kernel computes the Gram matrix of the propagation kernel for 
% a set of graphs
%
% Defined in: 
% [1] 'Efficient Graph Kernels by Randomization', 
%      M. Neumann, N. Patricia, R. Garnett, K. Kersting, ECML/PKDD 2012
%
% [ K, runtime ] = propagation_kernel(Graphs,T,w,TV,discrete)
%
% Input variables:
%   Graphs         input graph set saved in a matlab struct as follows:
%                     Graphs(i).am should be a binary n x n (symmetric) adjacency matrix, 
%                                  where n is the umber of nodes
%                     Graphs(i).nl.values contains discrete node labels (stored as an n x 1 vector of integers)
%                     Graphs(i).nl.vecvalues contains continuous d-dimensional vector node attributes (stored as a n x d double matrix)
%   T              a natural number: number of iterations of kernel
%   w              a positive number defining the width for the hash function (see [1] for details)
%   TV             a boolean: if 1 total variation distance is used,
%                             otherwise the Hellinger distance is used (see [1] for details)
%   discrete       a boolean: if 1 uses discrete (nl.values) labels, 
%                             otherwise continous labels (nl.vecvalues) are used
%
% Output variables:
%   K         T-element cell array of NxN Gram matrices for each 
%             iteration, where N is the number of graphs
%   runtime   Total cputime used by the kernel computation
% 
% This code makes use of the parallel computing toolbox if you make N > 1
% cores available prior to running this code by executing
%    'matlabpool local N', where N is your wanted number of cores
%
% This instance of the PROP kernel is different from those used in [1], as the continuous 
% label distributions described in [1] were replaced by continuous-valued features as indicated 
% in [1, page 4, footnote 3]. In personal communication, Marion Neumann has explained that 
% propagating continuous-valued features, as in this version, leads to a smoothing, or averaging, 
% of the continuous node features, leading to poor ability to capture graph structure. 
% This affects our implementation of PROP-diff. See the enclosed commentary for details.

% Copyright (C) 2013 Niklas Kasenburg
% 
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
%
% Author: Niklas Kasenburg <niklas dot kasenburg at tuebingen dot mpg dot de>
% 
% 2014-03-27 Niklas Kasenburg <niklas dot kasenburg at tuebingen dot mpg dot de>
%
% * 2014-03-27 : Error fixed in computing the inverse diagonal degree matrix D^-1 (see [1] for details)
%                * previous version: D = diag(ones(1,size(Graphs(i).am,1)) / sum(Graphs(i).am))
%                * previous version returns a number for D instead of a matrix (relates to using the wrong division) 
%                * previous version used right matrix division instead of elementwise division 
%                  (see Matlab functions 'mrdivide' and 'rdivide' for details)
% * 2013-10-22 : Initial revision


function [ K, runtime ] = propagation_kernel(Graphs,T,w,TV,discrete)

% number of graphs
N = size(Graphs,2);
% dimensionality of vector labels
if(discrete)
    d = 1;
else
    d = size(Graphs(1).nl.vecvalues,2);
end

% initialize output
K = cell(1,T);
K{1} = zeros(N);

time=cputime; % for measuring runtime

% initialize variables
n_nodes = 0;
labels = cell(1,N);
trans = cell(1,N);
% compute n_nodes, the total number of nodes in the dataset
% compute diffusion scheme transition matrix (see [1] for details)
% extract label matrix for each graph
for i=1:N
  n_nodes = n_nodes+size(Graphs(i).am,1);
  if(discrete)
    labels{i} = Graphs(i).nl.values; %does not use a label distribution (only the same for first iteration)
  else
    labels{i} = Graphs(i).nl.vecvalues;
  end
  tmp = 1 ./ sum(Graphs(i).am);
  tmp(isinf(tmp)) = 0;
  tmp(isnan(tmp)) = 0;
  trans{i} = diag(tmp);
  trans{i} = trans{i} * Graphs(i).am;
end

for t=1:T
    % each column i of phi will be the explicit feature representation for the graph i
    % in the extreme case each node can have a unique label => number of features = n_nodes 
    phi=sparse(n_nodes,N); 
    rng(t); % random seed used for comparable studies
    if(TV)
        % cauchy distributed ( 1/pi * 1/(1+x^2) )  random vector
        v = tan(pi*(rand(1,d)-0.5));
    else
        % standard normal distributed random vector
        v = normrnd(0,1,1,d);
    end
    % uniformly chosen ([0,w]) random value
    b = w*rand(1,1);
    % for mapping hash_values to indices in [1,n_nodes]
    hash_map = containers.Map('KeyType','int32','ValueType','int32');
    mapped_index = 1;
    % update feature-vector
    for i=1:N
        % compute hash values
        if(TV)
            hash = floor(( labels{1,i}*v' + b) / w)';
        else
            hash = floor(( sqrt(labels{1,i})*v' + b) / w)';
        end
        % transform hash values to indices 
        temp = zeros(size(hash));
        hash = int32(hash);
        for k=1:size(hash,2)
            cur_key = hash(k);
            if isKey(hash_map, cur_key)
                temp(k) = hash_map(cur_key);
            else
                hash_map(cur_key) = mapped_index;
                temp(k) = hash_map(cur_key);
                mapped_index = mapped_index + 1;
            end
        end
        % appending n_nodes guarantees that each vector has length n_nodes
        % count occurence of indices (labels)
        temp = accumarray([temp n_nodes]',1);
        temp(n_nodes) = temp(n_nodes)-1; % remove previously added n_nodes
        % update phi
        phi(:,i) = phi(:,i) + temp;
        
        % update labels (diffusion scheme)
        labels{1,i} = trans{1,i} * labels{1,i};
        
    end
    % update kernel
    if t == 1
        K{t} = full(phi'*phi);
    else
        K{t}=K{t-1}+full(phi'*phi);
    end
end

runtime=cputime-time; % computation time of K
disp(['Kernel computation finished after ', num2str(runtime), 'sec']);
end
