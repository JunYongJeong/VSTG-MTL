function [beta_opt,fun] = ksupport_FISTA_logistic(X,y,lambda,opts)
% Last modified on May 9, 2018


%% initilziation


if isfield(opts, 'init_w')
    w_current=opts.init_w;
    w_old = opts.init_w;    
else    
    w_current = zeros(size(X,2),1);
    w_old = w_current;
end

N  = length(y);

k=opts.k;
t=1;
t_old=0;

iter=0;
fun=[];

%% pre-compute

if isfield(opts, 'L')
    L = opts.L;
else
    [~,~,H] = Loss_logistic(X,y,w_current);
    L  =eigs(H,1)*0.96;
end

is_contin=1;
if max(abs(X))==0
    beta_opt = zeros(size(X,2),1);
    fun=0;
    is_contin=0;
end


%% main loop
while iter<opts.max_iter_sub & is_contin
    alpha = (t_old-1)/t;
    beta_s = (1+alpha)*w_current - alpha*w_old;
    [~,grad,~] = Loss_logistic(X,y,beta_s);
        
    w_old =w_current;
    [~,~,H] = Loss_logistic(X,y,w_current);
    L  =eigs(H,1)*0.96;

    w_current = prox_ksupport(beta_s - grad/L,k,2*lambda/L);

    fun = cat(1,fun, Loss_logistic(X,y,w_current) + lambda*norm_overlap(w_current,k)^2);
    

    if iter>=2 & fun(end-1) - fun(end) <=opts.rel_tol * fun(end-1)
        break;
    end
    
    iter=iter+1;
    t_old=t;
    t=0.5 * (1+(1+4*t^2)^0.5);

end
beta_opt = w_current;

end

