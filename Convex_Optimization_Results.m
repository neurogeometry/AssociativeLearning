% This function generates the numerical results for the biologically constrained,
% single-neuron associative learning model described in the manuscript.
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

function [W,C] = Convex_Optimization_Results(X,Xp,N,N_inh,w,kappa)

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

g=zeros(N,1);
g(1:N_inh) = -1;
g(N_inh+1:end) = 1;

delta = 10^-10;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)));
opts = optimoptions('linprog','Algorithm','interior-point','Display','off','MaxIterations',10^4);

m = length(Xp);
Xp = 2*Xp-1;

AA = [-((ones(N,1)*Xp).*X)', diag(-ones(m,1))];
BB = [-diag(g),zeros(N,m)];
CC = [zeros(m,N),diag(-ones(m,1))];
A = [AA;BB;CC];
b = [-Xp'-kappa;zeros(N+m,1)];
f = [zeros(N,1);ones(m,1)];
Aeq = [-ones(1,N_inh),ones(1,N-N_inh),zeros(1,m)];
beq = N*w;

% solution of Eqs. (39) of Supplementary Information with h=1
[SV,fval,exitflag] = linprog(f,A,b,Aeq,beq,[],[],opts);

if exitflag~=1
    error(['linprog did not converge to a solution, exitflag = ', num2str(exitflag)])
end

% if the above problem is feasible, look for the solution that minimizes the l-2 norm of the connection weights, Eqs. (38)
if fval<delta
    disp('Problem is feasible.')
    opts = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off','TolCon',10^-12,'TolX',10^-12,'TolFun',10^-12,'MaxIter',10^4);
    
    AA = -((ones(N,1)*Xp).*X)'; %association
    BB = -diag(g);%sign constrainted
    A = [AA;BB];
    b = [-Xp'-kappa;zeros(N,1)];
    H = diag(ones(N,1));
    Aeq = [-ones(1,N_inh),ones(1,N-N_inh)];
    beq = N*w;
    
    [SV,~,exitflag]=quadprog(H,[],A,b,Aeq,beq,[],[],SV(1:N),opts);
    if exitflag~=1
        error(['quadprog did not converge to a solution, exitflag = ', num2str(exitflag)])
    end
else
    disp('Problem is non-feasible.')
end

W = SV(1:N);
C = sum(sum(W'*((ones(N,1)*Xp).*X)-Xp > kappa-delta))/m;
