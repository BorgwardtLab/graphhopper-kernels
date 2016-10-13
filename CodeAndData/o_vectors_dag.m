% o_vectors_dag computes, given a single-source shortest path DAG adjacency 
% matrix G, the set of occurrence vectors for G.
%
% Defined in the paper 'Scalable kernels for graphs with continuous 
% attributes (A. Feragen, N. Kasenburg, J. Petersen, M. de Bruijne, 
% K. Borgwardt), Neural Information Processing Systems (NIPS) 2013'.
%
% occ = o_vectors_dag(G, source_index)
%
% Input variables:
%  * DAG G induced from a gappy tree where the indexing of nodes gives a 
%    breadth first order of the corresponding original graph
%  * source_index; the index of the source node
%
% Output variables:
%   A n x d descendant matrix occ, where n = size(G, 1) loops through the 
%   nodes of G, and d = diameter of G. The rows of the occ matrix will be 
%   padded with zeros on the right.   

% Copyright (C) 2013 Aasa Feragen
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
% Author: Aasa Feragen <aasa at diku dot dk>
% 
% 2013-10-22 Aasa Feragen <aasa at diku dot dk>
% * Initial revision

function occ = o_vectors_dag(G, shortestpath_dists)
   
    G = sparse(G);
    % Use sparse representation of DAG for speed
    
    %----------------------------------------------------------------------
    % Compute basic variables
        
    dag_size = size(G, 1);
    DAG_gen_vector = shortestpath_dists + 1;
    
    % This only works when the DAG is a shortest path DAG on an unweighted graph
    [~, gen_sorted] = sort(DAG_gen_vector);
    [~, re_sorted] = sort(gen_sorted);
    sortedG = G(gen_sorted, gen_sorted);
    delta = max(DAG_gen_vector);
    
    % Initialize: 
    % For a node v at generation i in the tree, give it the vector 
    % [0 0 ... 1 ... 0] of length h_tree with the 1 at the ith place.

    occ = zeros(dag_size, delta);
    occ(1,1) = 1;    
    
    for i = 1 : dag_size
        edges_starting_at_ith = find(squeeze(sortedG(i,:)) == 1);
        occ(edges_starting_at_ith, :) = occ(edges_starting_at_ith, :) + repmat([0, occ(i,1:end-1)], numel(edges_starting_at_ith), 1);
        clear edges_starting_at_ith
    end
    
    occ = occ(re_sorted, :);
end