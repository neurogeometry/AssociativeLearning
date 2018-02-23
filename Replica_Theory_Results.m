% This function generates the replica theoretical results for the biologically constrained,
% single-neuron associative learning model described in the manuscript.
% The function produces the results for both the associative and balanced models based on
% Eqs. 36 of SI. These models include:
% (1) excitatory and inhibitory inputs with sign-constrained weights
% (2) l-1 norm constraint on input weights
% (3) constant threshold

% INPUT PARAMETERS:
% N: total number of inputs
% Ninh: number of inhibitory inputs
% w: l-1 norm (i.e. average absolute connection weight)
% f: firing probability
% kappa: robustness parameter

% OUTPUTS PARAMETERS:
% alpha: memory storage capacity
% pcon_exc and pcon_inh: excitatory and inhibitory connection probabilities
% J_exc and J_inh: means of non-zero excitatory and inhibitory connection weights
% std_exc and std_inh: standard deviations of non-zero excitatory and inhibitory connection weights

function Replica_Theory_Results(N,N_inh,w,f,kappa,model)


% VALIDATION OF PARAMETERS 
assert(N>0,'N must be positive')
assert((N_inh>=0 & N_inh<N),'Ninh must be in the [0 N) range')
assert(w>1/f , 'w must be greater than 1/f')
assert((f>0 & f<1),'f must be in the (0 1) range')
assert(kappa>0,'kappa must be greater than zero')

alpha=[]; pcon_exc=[]; pcon_inh=[]; J_exc=[]; J_inh=[]; std_exc=[]; std_inh=[];

E = @(x) (1+erf(x))/2;
F = @(x) exp(-x.^2)./pi^0.5+x.*(1+erf(x));
D = @(x) x.*F(x)+E(x);

finh = N_inh/N;
rou = kappa/w/sqrt(f*(1-f));

options = optimset('Display','off','MaxIter',10^3,'MaxFunEvals',10^3,'TolX',10^-12,'TolFun',10^-12);
solution_found=false;
exitflag=0;
count=0;
x=[2.*(rand(1,4)-1),10*rand];

switch model
    case 'associative'
        S1 = @(x) [f.*F(x(2))-(1-f).*F(x(1));...
            (1-finh).*F(x(4))+finh.*F(x(3))-sqrt(2)./x(5);...
            (1-finh).*F(x(4))-finh.*F(x(3))-sqrt(2)./x(5)/w/f;...
            ((1-finh).*D(x(4))+finh.*D(x(3))).*(x(1)+x(2)).^2.*x(5).^2-2.*rou^2;...
            sqrt(2)*rou^2*(f.*F(x(2))+(1-f).*F(x(1)))./(f.*E(x(2))+(1-f).*E(x(1)))-x(5).*(x(1)+x(2)).*((x(3)-x(4))/w/f-(x(3)+x(4)))];
        
        while ~(exitflag==1 && (x(1)+x(2))>0 && x(5)>0) && count<=500
            x=[2.*(rand(1,4)-1),10*rand];
            count=count+1;
            [x,~,exitflag] = fsolve(S1, x, options);
            if exitflag==1 && (x(1)+x(2))>0 && x(5)>0
                solution_found=true;
                alpha = 2*rou^2*(f*D(x(2)) + (1-f)*D(x(1)))/(f*E(x(2)) + (1-f)*E(x(1))).^2/(x(1)+x(2)).^2/x(5)^2;
                pcon_exc = E(x(4));
                pcon_inh = E(x(3));
                J_exc = w*x(5)*F(x(4))/sqrt(2)/E(x(4));
                J_inh = w*x(5)*F(x(3))/sqrt(2)/E(x(3));
                std_exc = J_exc*sqrt(2*D(x(4))*E(x(4))/F(x(4))^2-1);
                std_inh = J_inh*sqrt(2*D(x(3))*E(x(3))/F(x(3))^2-1);
            end
        end
    case 'balanced'
        S2 = @(x) [f.*F(x(2))-(1-f).*F(x(1));...
            (1-finh).*F(x(4))+finh.*F(x(3))-sqrt(2)./x(5);...
            (1-finh).*F(x(4))-finh.*F(x(3));...
            ((1-finh).*D(x(4))+finh.*D(x(3))).*(x(1)+x(2)).^2.*x(5).^2-2.*rou^2;...
            sqrt(2)*rou^2*(f.*F(x(2))+(1-f).*F(x(1)))./(f.*E(x(2))+(1-f).*E(x(1)))-x(5).*(x(1)+x(2)).*(-(x(3)+x(4)))];
        
        while ~(exitflag==1 && (x(1)+x(2))>0 && x(5)>0) && count<=500
            x=[2.*(rand(1,4)-1),10*rand];
            count=count+1;
            [x,~,exitflag] = fsolve(S2, x, options);
            if exitflag==1 && (x(1)+x(2))>0 && x(5)>0
                solution_found=true;
                alpha = 2*rou^2*(f*D(x(2)) + (1-f)*D(x(1)))/(f*E(x(2)) + (1-f)*E(x(1))).^2/(x(1)+x(2)).^2/x(5)^2;
                pcon_exc = E(x(4));
                pcon_inh = E(x(3));
                J_exc = w.*x(5)*F(x(4))/sqrt(2)/E(x(4));
                J_inh = w.*x(5)*F(x(3))/sqrt(2)/E(x(3));
                std_exc = J_exc*sqrt(2*D(x(4))*E(x(4))/F(x(4))^2-1);
                std_inh = J_inh*sqrt(2*D(x(3))*E(x(3))/F(x(3))^2-1);
            end
        end
    otherwise
        error('Unexpected model, use associative or balanced.')
end

if solution_found==true
    disp(['Critical capacity:                                 ', num2str(alpha)])
    disp(['Excitatory connection probability:                 ', num2str(pcon_exc')])
    disp(['Inhibitory connection probability:                 ', num2str(pcon_inh')])
    disp(['Average non-zero excitatory connection weight:     ', num2str(J_exc')])
    disp(['Average non-zero inhibitory connection weight:     ', num2str(J_inh')])
    disp(['Standard deviation of non-zero excitatory weights: ', num2str(std_exc')])
    disp(['Standard deviation of non-zero inhibitory weights: ', num2str(std_inh')])
else
    disp('Solution not found.')
end
