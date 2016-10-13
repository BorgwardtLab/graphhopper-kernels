% GraphHopper_dataset computes the Gram matrix of the GraphHopper kernel on 
% a set of graphs
%
% Defined in the paper 'Scalable kernels for graphs with continuous 
% attributes (A. Feragen, N. Kasenburg, J. Petersen, M. de Bruijne, 
% K. Borgwardt), Neural Information Processing Systems (NIPS) 2013'.
%
% [K, comp_time] = GraphHopper_dataset(Graphs, node_kernel_type, mu, vecvalues)
%
% Input variables:
%   Graphs         input graph set saved in a matlab struct as follows:
%                     Graphs(i).am should be a binary N x N (symmetric) adjacency matrix
%                     Graphs(i).nl.values contains discrete node labels (stored as an N x 1 vector of integers)
%                     Graphs(i).nl.vecvalues contains continuous m-dimensional vector node attributes (stored as a N x m double matrix)
%   kerneltype     'linear',  'gaussian', 'diractimesgaussian', 'dirac', 
%                  'bridge' ('dirac' uses only discrete node labels)
%   mu             mu parameter for the gaussian node kernel on the 3d location node attributes
%   vecvalues      binary parameter.
%                     0 = Use only scalar node attributes stored in Graphs.nl.values
%                     1 = Use only vector valued node attributes stored in Graphs.nl.vecvalues
%
% Output variables:
%   K           N x N Gram matrix, where N is the number of graphs
%   comp_time   Total cputime used by the kernel computation
% 
% This code makes use of the parallel computing toolbox if you make N > 1
% cores available prior to running this code by executing
%    'matlabpool local N', where N is your wanted number of cores

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
% * Version 1.3 (14.1.2016): Updated to avoid double calculations and numerical imprecision resulting in unsymmetric kernel matrices (Thank you to Marco Alvarez who spotted the bug!).

function [K, comp_time] = GraphHopper_dataset(Graphs, node_kernel_type, mu, vecvalues)
       
    % ---------------------------------------------------------------------
    
    % Set default settings: Using vector-valued features; mu = 1, linear
    % node kernel
    
    if nargin < 4
        vecvalues = 1;
    end
    if nargin < 3
        mu = 1;
    end
    if nargin < 2
        node_kernel_type = 'linear';
    end
       
             
    %---------------------------------------------------------------------
    
    % First, compute basic parameters

    num_graphs = numel (Graphs);
    M_mat{num_graphs} = [];      
    diam = zeros(1, num_graphs);
    graph_size = zeros(1, num_graphs);
        
    for i = 1 : num_graphs                
        M = Graphs(i).am;   
        M = graphallshortestpaths(sparse(M));
        path_edge_nr = M(:);
        % Assume that the adjacency matrix is binary (could easily be extended)
        diam(i) = max(path_edge_nr(path_edge_nr< Inf));
        graph_size(i) = size(M,1);
    end
    max_diam =  max(diam) + 1;
    
    %----------------------------------------------------------------------
    
    % First preprocessing step:
    %   * Compute descendant vectors for all nodes in all rooted DAGs
    
    t1 = cputime;
    for i = 1 : num_graphs
        display(['Preprocessing graph ', num2str(i)]);
        AM = Graphs(i).am;    
        node_nr = graph_size(i);
        des = zeros(node_nr, node_nr, max_diam);
        occ = zeros(node_nr, node_nr, max_diam);
        for j = 1 : node_nr
            A = 0*AM;                                                       % A is going to be the DAG adjacency matrix
            [D, ~, p] = graphshortestpath(sparse(AM), j);                        % Single-source shortest path from node j            
            conn_comp = find(D < Inf);                                      % Restrict to the connected component of node j
            newind_j = find(conn_comp == j);                                % Extract the indices of nodes in the connected component of node j
            A_cc = A(conn_comp, conn_comp);                                 % To-be DAG adjacency matrix of connected component of node j
            AM_cc = AM(conn_comp, conn_comp);                               % Adjacency matrix of connected component of node j
            % FIX THIS TO AVOID SECOND DIJKSTRA
            D_cc = D(conn_comp);
            conn_comp_converter = zeros(size(A, 1), 1);
            for k = 1 : numel(conn_comp)
                conn_comp_converter(conn_comp(k)) = k;
            end
            conn_comp_converter = vertcat(0, conn_comp_converter);
            p_cc = conn_comp_converter(p(conn_comp) + 1);
%             p_cc = find(conn_comp == p(conn_comp));
            % OBS! THIS PART DOES NOT WORK YET!
%             [D_cc, p_cc] = shortest_paths(sparse(AM_cc), newind_j);         % Single-source shortest path from node j in connected component
            conncomp_node_nr = size(A_cc, 1);                               % Number of nodes in connected component of node j
            for v = 1 : conncomp_node_nr
                if p_cc(v) > 0
                    A_cc(p_cc(v), v) = 1;                                   % Generate A_cc by adding directed edges of form (parent(v), v)
                end
                v_dist = D_cc(v);                                           % Distance from v to j
                v_nbs = find(AM_cc(v,:) > 0);                               % All neighbors of v in the undirected graph
                v_nbs_dists = D_cc(v_nbs);                                  % Distances of neighbors of v to j
                v_parents = v_nbs(v_nbs_dists == v_dist - 1);               % All neighbors of v in undirected graph who are one step closer to j than v is; i.e. SP-DAG parents
                A_cc(v_parents, v) = 1;                                     % Add SP-DAG parents to A_cc
            end            
            des1 = d_vectors_dag(A_cc, D_cc);                               % Computes the descendants vectors d_j(v) for all v in the connected component            
            occ1 = o_vectors_dag(A_cc, D_cc);                               % Computes the occurrence vectors o_j(v) for all v in the connected component
            if numel(des1) == 1 && j == 1                
                des(j, 1, 1) = des1;
                occ(j, 1, 1) = occ1;
            else
                for v = 1: size(des1, 1)
                    for l = 1 : size(des1, 2)
                        des(j,conn_comp(v),l) = des1(v,l);                  % Convert back to the indices of the original graph
                    end
                end
                for v = 1: size(occ1, 1)
                    for l = 1 : size(occ1, 2)
                        occ(j,conn_comp(v),l) = occ1(v,l);                  % Convert back to the indices of the original graph
                    end
                end
            end                            
        end
        
        % Now:
        %   * des is a node_nr x node_nr x max_diam matrix with des(j, k, :)
        %     the descendant vector d_j(k)
        %   * occ is a node_nr x node_nr x max_diam matrix with occ(j, k, :)
        %     the occurence vector o_j(k)
        
        M = zeros(graph_size(i), max_diam, max_diam);
        for j = 1 : graph_size(i)                                           % j loops through choices of root                         
            des_mat_j_root = squeeze(des(j,:,:));
            occ_mat_j_root = squeeze(occ(j,:,:));
            for v = 1 : graph_size(i)                                       % k loops through nodes                
                for a = 1 : max_diam
                    for b = a : max_diam
                         M(v, a, b) = M(v, a, b) + des_mat_j_root(v, b - a + 1)*occ_mat_j_root(v, a);   % M(v,:,:) is M(v); a = node coordinate in path, b = path length                        
                    end
                end
            end
        end
        M_mat{i} = M; 
        clear des des1 k_dist k_nbs k_nbs_dists k_parents
    end
    
    %----------------------------------------------------------------------
    % Step 4: compute the kernel
    
    K = zeros(num_graphs);
    if strcmp(node_kernel_type, 'linear')        
        for i = 1 : num_graphs        
            display(['computing weight matrix ', num2str(i)])
            M_i = M_mat{i};
            M_i = reshape(M_i, graph_size(i), max_diam^2);
            if vecvalues == 1
                NA_i = Graphs(i).nl.vecvalues;
            else
                NA_i = Graphs(i).nl.values;
            end
            for j = 1 : num_graphs
                M_j = M_mat{j};
                M_j = reshape(M_j, graph_size(j), max_diam^2);
                weight_matrix = M_i*M_j';
                if vecvalues == 1
                    NA_j = Graphs(j).nl.vecvalues;
                else
                    NA_j = Graphs(j).nl.values;
                end
                NA_linear_kernel = NA_i * NA_j';
                K(i,j) =  weight_matrix(:).' * NA_linear_kernel(:);
            end
            for j = i : num_graphs
                K(j,i) = K(i,j);
            end        
        end
    elseif strcmp(node_kernel_type, 'gaussian')        
        for i = 1 : num_graphs        
            display(['computing weight matrix ', num2str(i)])
            M_i = M_mat{i};
            M_i = reshape(M_i, graph_size(i), max_diam^2);
            if vecvalues == 1
                NA_i = Graphs(i).nl.vecvalues;
            else
                NA_i = Graphs(i).nl.values;
            end
            norm2_i = sum (NA_i .* NA_i, 2);
            for j = 1 : num_graphs                
                M_j = M_mat{j};
                M_j = reshape(M_j, graph_size(j), max_diam^2);
                weight_matrix = M_i*M_j';
                if vecvalues == 1
                    NA_j = Graphs(j).nl.vecvalues;
                else
                    NA_j = Graphs(j).nl.values;
                end
                norm2_j = sum (NA_j .* NA_j, 2);
                NA_linear_kernel = NA_i * NA_j';
                NA_squared_distmatrix = bsxfun (@plus, bsxfun (@plus, -2 * NA_linear_kernel, norm2_i), norm2_j.');
                K_nodepair = exp(-mu*NA_squared_distmatrix);
                K(i,j) =  weight_matrix(:).' * K_nodepair(:);
            end
            for j = i : num_graphs
                K(j,i) = K(i,j);
            end        
        end
    elseif strcmp(node_kernel_type, 'diractimesgaussian')
        for i = 1 : num_graphs        
            display(['computing weight matrix ', num2str(i)])
            M_i = M_mat{i};
            M_i = reshape(M_i, graph_size(i), max_diam^2);
            if vecvalues == 1
                NA_i = Graphs(i).nl.vecvalues;
                discrete_i = Graphs(i).nl.values;
            else
                NA_i = Graphs(i).nl.values;
                discrete_i = Graphs(i).nl.values;
            end
            norm2_i = sum (NA_i' .* NA_i', 1);
            for j = 1 : num_graphs
                M_j = M_mat{j};
                M_j = reshape(M_j, graph_size(j), max_diam^2);
                weight_matrix = M_i*M_j';
                if vecvalues == 1
                    NA_j = Graphs(j).nl.vecvalues;
                    discrete_j = Graphs(j).nl.values;
                else
                    NA_j = Graphs(j).nl.values;
                    discrete_j = Graphs(j).nl.values;
                end
                norm2_j = sum (NA_j' .* NA_j', 1);
                NA_linear_kernel = NA_i * NA_j';
                NA_squared_distmatrix = bsxfun (@plus, bsxfun (@plus, -2 * NA_linear_kernel, norm2_i'), norm2_j);
                K_cont_nodepair = exp(-mu*NA_squared_distmatrix);
                
                K_discrete_nodepair = zeros(numel(discrete_i), numel(discrete_j));
                for v = 1 : numel(discrete_i)
                    ones = find(discrete_j == discrete_i(v));
                    K_discrete_nodepair(v, ones) = 1;
                end  
                K_nodepair = K_discrete_nodepair.*K_cont_nodepair;
                K(i,j) =  weight_matrix(:).' * K_nodepair(:);
		K(j,i) = K(i,j);
            end   
        end        
    elseif strcmp(node_kernel_type, 'dirac')
        for i = 1 : num_graphs        
            display(['computing weight matrix ', num2str(i)])
            M_i = M_mat{i};
            M_i = reshape(M_i, graph_size(i), max_diam^2);
            NA_i = Graphs(i).nl.values;
            for j = 1 : num_graphs
                M_j = M_mat{j};
                M_j = reshape(M_j, graph_size(j), max_diam^2);
                weight_matrix = M_i*M_j';
                NA_j = Graphs(j).nl.values;
                K_nodepair = zeros(numel(NA_i), numel(NA_j));
                for v = 1 : numel(NA_i)
                    ones = find(NA_j == NA_i(v));
                    K_nodepair(v, ones) = 1;
                    clear ones
                end                    
                K(i,j) =  weight_matrix(:).' * K_nodepair(:);
            end
            for j = i : num_graphs
                K(j,i) = K(i,j);
            end        
        end  
    elseif strcmp(node_kernel_type, 'bridge')
        % set c parameter:
        c = 4;
        for i = 1 : num_graphs        
            display(['computing weight matrix ', num2str(i)])
            M_i = M_mat{i};
            M_i = reshape(M_i, graph_size(i), max_diam^2);
            if vecvalues == 1
                NA_i = Graphs(i).nl.vecvalues;
            else
                NA_i = Graphs(i).nl.values;
            end                        
            for j = 1 : num_graphs
                M_j = M_mat{j};
                M_j = reshape(M_j, graph_size(j), max_diam^2);
                weight_matrix = M_i*M_j';
                if vecvalues == 1
                    NA_j = Graphs(j).nl.vecvalues;
                else
                    NA_j = Graphs(j).nl.values;
                end
                NAs = vertcat(NA_i, NA_j);
                NAs_linear_kernel = NAs * NAs';
                NAs_distances = kernelmatrix2distmatrix(NAs_linear_kernel);
                NA_i_NA_j_distances = NAs_distances(1:size(NA_i, 1), size(NA_i, 1) + 1 : end);
                K_nodepair = (1/c)*max(0, c-NA_i_NA_j_distances);
                K(i,j) =  weight_matrix(:).' * K_nodepair(:);
            end
            for j = i : num_graphs
                K(j,i) = K(i,j);
            end        
        end        
    end
    
    t2 = cputime;
    comp_time = t2-t1;
end
