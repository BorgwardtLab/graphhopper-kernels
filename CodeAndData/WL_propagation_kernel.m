% WL_propagation_kernel computes the Gram matrix of the Weisfeiler-Lehman variant 
% of thepropagation kernel for a set of graphs
%
% Defined in: 
% [1] 'Efficient Graph Kernels by Randomization', 
%      M. Neumann, N. Patricia, R. Garnett, K. Kersting, ECML/PKDD 2012
% [2] 'Weisfeiler-Lehman graph kernels'
%      N. Shervashidze, P. Schweitzer, E.J. van Leeuwen, K. Mehlhorn, and K.M. Borgwardt, JMLR, 2011
%
% [ K, runtime ] = WL_propagation_kernel(Graphs,T,w,TV,discrete)
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
%   discret        a boolean: if uses discrete (nl.values) labels, 
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
% Copyright (C) 2013 Niklas Kasenburg and Nino Shervashidze
% 
% This program includes code from an implementation of the Weifeiler-Lehman Kernel [2] 
% written by Nino Shervashidze
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
% 
% Author: Niklas Kasenburg <niklas dot kasenburg at tuebingen dot mpg dot de>
% 
% 2013-10-22 Niklas Kasenburg <niklas dot kasenburg at tuebingen dot mpg dot de>
% * Initial revision

function [ K, runtime ] = WL_propagation_kernel(Graphs,T,w,TV,discrete)

% number of graphs
N = size(Graphs,2);
% dimensionality of vector labels
if(discrete)
    d = 1;
else
    d = size(Graphs(1).nl.vecvalues,2);
end
% initialize kernel
K = cell(1,T);
K{1} = zeros(N);

time=cputime; % for measuring runtime

n_nodes=0;
labels = cell(1,N);
new_labels = cell(1,N);
% compute n_nodes, the total number of nodes in the dataset
% construct label matrix for each graph
for i=1:N
  n_nodes=n_nodes+size(Graphs(i).am,1);
  if(discrete)
    labels{i} = Graphs(i).nl.values;
    new_labels{i}=zeros(size(Graphs(i).nl.values,1),1,'uint32');
  else
    labels{i} = Graphs(i).nl.vecvalues;
    new_labels{i}=zeros(size(Graphs(i).nl.vecvalues,1),1,'uint32');
  end
end


for t=1:T
    % each column i of phi will be the explicit feature representation for the graph i
    % in the extreme case each not can have unique label => number of features = n_nodes 
    phi=sparse(n_nodes,N);
    rng(t); % random seed used for comparable studies
    if(TV)
        %cauchy distributed ( 1/pi * 1/(1+x^2) )  random vector
        v = tan(pi*(rand(1,d)-0.5));
    else
        %standard normal distributed random vector
        v = normrnd(0,1,1,d);
    end
    % uniformly chosen ([0,w]) random value
    b = w*rand(1,1);
    % for mapping hash_values to indices in [1,n_nodes]
    hash_map = containers.Map('KeyType','int32','ValueType','uint32');
    mapped_index = uint32(1);
    % discrete labels
    hashed_labels = cell(1,N);
    
    % hash labels to discrete values before first iteration 
    if t == 1 
    % bin labels according to their hash values
        for i=1:N
            % compute hash values
            if(TV)
                hash = floor(( labels{1,i}*v' + b) / w)';
            else
                hash = floor(( sqrt(labels{1,i})*v' + b) / w)';
            end
            %transform hash values to indices (discretize labels) 
            temp = zeros(size(hash),'uint32');
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
            % set new labels
            hashed_labels{1,i} = temp';
        end
    % use Weifeiler-Lehman updated labels in all other itaration    
    else
        hashed_labels = new_labels;
    end
    
    % WL signature computation and label update (includes large parts written by Nino Shervashidze)
    
    % create an empty lookup table
    label_lookup=containers.Map();
    label_counter= uint32(1);
  
    for j=1:N
        list = Graphs(j).al;  
        for v = 1:length(list)
            % use original labels for first iteration
            if (t==1)
                long_label_string=num2str(hashed_labels{1,j}(v));
            else
                % form a multiset label of the node v of the i'th graph
                % and convert it to a string
                long_label=[hashed_labels{1,j}(v), sort(hashed_labels{1,j}(list{v}))'];
                long_label_2bytes=typecast(long_label,'uint32');
                long_label_string=char(long_label_2bytes);
            end
            % if the multiset label has not yet occurred, add it to the
            % lookup table and assign a number to it
            if ~isKey(label_lookup, long_label_string)
                label_lookup(long_label_string)=label_counter;
                new_labels{1,j}(v)=label_counter;
                label_counter=label_counter+1;
            else
                new_labels{1,j}(v)=label_lookup(long_label_string);
            end
        end
    % update phi    
    aux = accumarray(new_labels{1,j}, ones(length(new_labels{1,j}),1));
    phi(new_labels{1,j},j)=phi(new_labels{1,j},j)+aux(new_labels{1,j});
    end
    L = label_counter-1;
    if (t==1)
        disp(['Number of original hashed labels: ',num2str(L)]);
    else
        disp(['Number of compressed labels: ',num2str(L)]);
    end
    % update kernel
    if(t==1)
        K{t} = full(phi'*phi);
    else
        K{t}=K{t-1}+full(phi'*phi);
    end
    
    % update labels
    for i=1:N
        labels{i} = double(new_labels{i});
    end
end

runtime=cputime-time; % computation time of K

end