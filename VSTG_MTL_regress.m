function [U,V,history] = VSTG_MTL_regress(X_cell,Y_cell,K,hyp, opts)
% Variable Selection and Task Grouping Multi-task Learning for regression
% Objective function

% description
% D variables and T tasks
% \sum_{j=1}^T ||y_j -X_j * U * v_j||_2^2/N_j
% + gamma_1 ||U||_1 + gamma2 ||U||_1,inf + mu * sum_{j=1}^T (||v_j||_k^{sp})^2
% Input arguments
% X_cell: T \times 1 input cell matrix, 
%         where the j-th component is X_j \in N_j * D
% Y_cell: T \times 1 output cell matrix,
%         where the j-th component is y_j \in N_j \in R^{N_j}
% K: rank of coeffcient matrix = # of latent bases
% hyp=[gamma1, gamma2, mu, k]: regularization parameters
% opts: opts.max_iter, opts.rel_tol,

% Last modified on May 10, 2018
%% Initialization 
T = length(Y_cell);
D = size(X_cell{1},2);
opts.k = hyp(4);
for task=1:T
    left = (X_cell{task}'*X_cell{task} + norm(hyp(1:3),2)+eye(D));
    right = X_cell{task}'*Y_cell{task};
    W_init(:,task) = left\right;        
end
[U_temp,Lambda,V_temp] = svds(W_init,K);
U = U_temp * sqrt(Lambda);
V = sqrt(Lambda) *V_temp';    
fun = Obj_val(U,V);
iter=1;
history.time_total_U(1)=0;
history.time_total_V(1)=0;

%% main

while iter<opts.max_iter
%     disp(iter)
    
    % update U
    tic
    [U, history_U] = update_MTL_U_ADMM_regress(U, V,X_cell,Y_cell,hyp(1:2),opts);
    history.time_iter_U(iter) = toc;
    history.time_total_U(iter) = history.time_total_U(max(1,iter-1)) + history.time_iter_U(iter);
    
    % update V       
    tic
    for task=1:T
        Z = X_cell{task}*U;
        N_task = length(Y_cell{task});
        opts.init_beta = V(:,task);
        
        V(:,task) = ksupport_FISTA_regress(Z,Y_cell{task}, hyp(3), opts);
    end      
    history.time_iter_V(iter) = toc;
    history.time_total_V(iter) = history.time_total_V(max(1,iter-1)) + history.time_iter_V(iter);
    
    f_new = Obj_val(U,V);
    fun = cat(1,fun, f_new);
    
    % stopping criteria 
    if iter>=2 && abs(fun(end)-fun(end-1))<= opts.rel_tol*fun(end-1)
        break;
    end       
    
    iter=iter+1;
end
history.fun = fun;



    %%      

    function val = Obj_val(U,V)
        
        val = hyp(1) * norm(U,1) + hyp(2)*sum(max(abs(U),[],2));
        for r=1:size(V,2)
            val = val + hyp(3) * norm_overlap(V(:,r),opts.k)^2;
        end
        
        
        for m2=1:T
            val = val + mean((Y_cell{m2} - X_cell{m2}*U*V(:,m2)).^2)/2;
        end             
    end

   




end

