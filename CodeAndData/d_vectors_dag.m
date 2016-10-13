% d_vectors_dag computes, given a single-source shortest path DAG adjacency 
% matrix G, the set of descendant vectors for G.
%
% Defined in the paper 'Scalable kernels for graphs with continuous 
% attributes (A. Feragen, N. Kasenburg, J. Petersen, M. de Bruijne, 
% K. Borgwardt), Neural Information Processing Systems (NIPS) 2013'.
%
% des = d_vectors_dag(G, source_index)
%
% Input variables:
%  * DAG G induced from a gappy tree where the indexing of nodes gives a 
%    breadth first order of the corresponding original graph
%  * source_index; the index of the source node
%
% Output variables:
%   A n x d descendant matrix des, where n = size(G, 1) loops through the 
%   nodes of G, and d = diameter of G. The rows of the des matrix will be 
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

function des = d_vectors_dag(G, shortestpath_dists)
    
    G = sparse(G);
    % Use sparse representation of DAG for speed
    
    %----------------------------------------------------------------------
    % Compute basic variables
        
    dag_size = size(G, 1);
    DAG_gen_vector = shortestpath_dists + 1;

    % This only works when the DAG is a shortest path DAG on an unweighted graph
    [~, gen_sorted] = sort(DAG_gen_vector);
    [~, re_sorted] = sort(gen_sorted);
    sorted_dirG = G(gen_sorted, gen_sorted);
    delta = max(DAG_gen_vector);
    
    % Initialize: 
    % For a node v at generation i in the tree, give it the vector 
    % [0 0 ... 1 ... 0] of length delta with the 1 at the ith place.
    
    des = zeros(dag_size, delta);
    des(:,1) = ones(1, dag_size);
    
    % Now use message-passing from the bottom of the DAG to add up the 
    % edges from each node. This is easy because the vertices in the DAG 
    % are depth-first ordered in the original tree; thus, we can just start
    % from the end of the DAG matrix.
    
    for i = 1 : dag_size
        edges_ending_at_ith_from_end = find(squeeze(sorted_dirG(:, dag_size - i + 1)) == 1);
        des(edges_ending_at_ith_from_end, :) = des(edges_ending_at_ith_from_end, :) + repmat([0, des(dag_size - i + 1, 1:end-1)], numel(edges_ending_at_ith_from_end), 1);
        clear edges_ending_at_ith_from_end
    end
    
    des = des(re_sorted, :);     
end
    