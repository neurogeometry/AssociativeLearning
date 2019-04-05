% This function generates the numerical results using a modified perceptron 
% learning rule for the biologically constrained, single-neuron associative
% learning model described in the manuscript.
% The function produces the results based on Eqs. 38, 39 of SI. The model includes:
% (1) excitatory and inhibitory inputs with sign-constrained weights
% (2) l-1 norm constraint on input weights
% (3) constant threshold, h=1

% This code works with MATLAB version R2017a or later

% INPUT PARAMETERS:
% X: input associations, N x number of associations
% Xp: outputs associations, 1 x number of associations
% N: total number of inputs
% N_inh: number of inhibitory inputs
% w: average absolute connection weight (l-1 norm constraint)
% kappa: robustness parameter

% OUTPUTS PARAMETERS:
% W: input weights, Nx1
% C: fraction of successfully learned associations

function [W,C] = Modified_Perceptron_Rule_Results(X,Xp,N,N_inh,w,kappa)

% VALIDATION OF PARAMETERS
assert(size(X,1)==N,'X must be N x number of associations, containing only zeros and ones')
assert(size(Xp,1)==1,'Xp must be 1 x number of associations, containing only zeros and ones')
assert(size(X,2)==size(Xp,2),'X and Xp must have the same second dimension size')
temp=unique(X(:));
if length(temp)==1
    assert((temp==0 || temp==1),'X must be N x number of associations, containing only zeros and ones')
elseif length(temp)==2
    assert(nnz(temp-[0;1])==0,'X must be N x number of associations, containing only zeros and ones')
else
    error('X must be N x number of associations, containing only zeros and ones')
end
temp=unique(Xp);
if length(temp)==1
    assert((temp==0 || temp==1),'Xp must be 1 x number of associations, containing only zeros and ones')
elseif length(temp)==2
    assert(nnz(temp-[0 1])==0,'Xp must be 1 x number of associations, containing only zeros and ones')
else
    error('Xp must be 1 x number of associations, containing only zeros and ones')
end
assert(N>0,'N must be a positive integer')
assert((N_inh>=0 & N_inh<N),'N_inh must be an integer in the [0 N) range')
assert(w>0, 'w must be positive')
assert(kappa>0,'kappa must be greater than zero')

Nsteps = 10^7; % maximum number of iteration steps
beta = 0.01; % learning rate

m = length(Xp);
Xp=2*Xp-1;
XX=X.*repmat(Xp./N,N,1);
W = w*[-ones(N_inh,1);ones(N-N_inh,1)];
g=[-ones(1,N_inh),ones(1,N-N_inh)]';
out=(kappa/N^0.5-W'*XX+Xp)>0;
bad_associations=nnz(out);
iteration=0;
while bad_associations>0 && iteration<Nsteps
    iteration=iteration+1;
    mu=find(out);
    mu = mu(randi(length(mu)));
    W = W + beta.* Xp(mu).*X(:,mu);
    W(W.*g<0)=0;
    x = (sum(W.*g) - N*w)/N;
    W = W - g.*x;
    W(W.*g<0)=0;
    out=(kappa/N^0.5-W'*XX+Xp)>0;
    bad_associations=nnz(out);
end
C = 1 - bad_associations/m;
end