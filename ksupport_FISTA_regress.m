function [beta_opt,fun] = ksupport_FISTA_regress(X,y,lambda,opts)
% Last modified on May 25, 2018


%% initilziation


if isfield(opts, 'init_beta')
    beta_current=opts.init_beta;
    beta_old = opts.init_beta;    
else    
    beta_current = zeros(size(X,2),1);
    beta_old = beta_current;
end

N  = length(y);

k=opts.k;
t=1;
t_old=0;

iter=0;
fun=[];
XtX = X'*X;
Xty = X'*y;


%% pre-compute 
if isfield(opts, 'L')
    L = opts.L;
else
    L= eigs(X'*X/N,1);
end

is_contin=1;
if max(abs(X))==0
    beta_opt = zeros(size(X,2),1);
    fun=0;
    is_contin=0;
end


%% main loop
while iter<opts.max_iter_fista & is_contin
    alpha = (t_old-1)/t;
    beta_s = (1+alpha)*beta_current - alpha*beta_old;
    grad = (XtX*beta_s - Xty)/N;
    
    beta_old =beta_current;
    beta_current = prox_ksupport(beta_s - grad/L,k,2*lambda/L);
    fun = cat(1,fun, mean((y-X*beta_current).^2)/2 + lambda*norm_overlap(beta_current,k)^2);
    
    if iter>=2 & fun(end-1) - fun(end) <=opts.rel_tol_fista * fun(end-1)
        break;
    end
    
    iter=iter+1;
    t_old=t;
    t=0.5 * (1+(1+4*t^2)^0.5);

end
beta_opt = beta_current;


end

