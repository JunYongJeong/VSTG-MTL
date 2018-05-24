function [ w, fun ] = L2_Newton_logistic(X,y, mu,opts)

% Last modified on May 9, 2018
%%
if isfield(opts, 'init_w')
    w=opts.init_w;
else    
    w = zeros(size(X,2),1);
end
N  = length(y);

t=1;
t_old=0;
fun =[];

if max(abs(X))==0
    w_opt = zeros(size(X,2),1);
    fun=0;
end

% 
% [N,dim_input] = size(X);
% iter=1;
% w = zeros(dim_input,1);
% fun=[];

%%
iter=1;
for iter=1:opts.max_iter
%     disp(iter)
    [~,g_loss, H_loss] = Loss_logistic(X,y,w);
    g = g_loss + mu * w;
    H = H_loss + mu* eye(length(w));
    d = H\g;
    w = w -d ;
    fun = cat(1, fun, Loss_logistic(X,y,w) + (mu/2) * norm(w,2)^2);

    
    if iter>=2 && fun(end-1)-fun(end) < opts.rel_tol*fun(end-1)
        break;
    end;

end



end

