function [ U_opt U_history ] = update_MTL_U_ADMM_regress(U_init, V,X_cell,Y_cell,hyp,opts)

% Last modified on May 25, 2018

%% initialization and pre-compute
M = length(Y_cell);
[D,K] = size(U_init);
U=U_init;
Z1=U;
Z2=U;
Z3=U;

L1=U;
L2=U;
L3=U;
iter=0;
  
rho =opts.rho;

% pre-compute
A = rho * eye(D*K);
B_pre = zeros(size(Z1));
for t=1:M
    N_t = size(X_cell{t},1);
    A = A + 1/N_t * kron(V(:,t)*V(:,t)', X_cell{t}'*X_cell{t});    
    B_pre = B_pre + 1/N_t * X_cell{t}'*Y_cell{t} * V(:,t)';
end

%% update
while iter<=opts.max_iter_ADMM
%     disp(iter)

    % update U
    U = (Z1+Z2+Z3 - L1-L2-L3)/3;
    
    % update auxiliary variables Z
    Z1_old = Z1;
    Z2_old= Z2;
    Z3_old = Z3;
    
    B = rho *(U(:) + L1(:)) + B_pre(:);
    
    % pre-compute A    
    Z1 = reshape(linsolve(A,B),size(Z1));    
    Z2 = prox_L11norm(U + L2, hyp(1)/rho);
    Z3 = prox_L1infnorm(U + L3, hyp(2)/rho);
    
    % update Lagrangian variables L
    
    L1 = L1+ U - Z1;
    L2 = L2 + U - Z2;
    L3 = L3 + U - Z3;
    
    iter=iter+1;
    U_history.fun(iter) = fun_eval(Z1,Z2,Z3);
    U_history.r_norm(iter) = norm([U-Z1,U-Z2,U-Z3],'fro');
    U_history.s_norm(iter) = rho * norm([Z1-Z1_old,Z2-Z2_old,Z3-Z3_old],'fro'); 
          
    if U_history.r_norm(iter) < opts.tol_ADMM && U_history.s_norm(iter)<= opts.tol_ADMM
        break;    
    end
    
end
U_opt = Z2;
U_opt(abs(U_opt)<10^-4)=0;


    %%   private function
    function val = fun_eval(Z1,Z2,Z3)
        val = hyp(1) * norm(Z2,1) + hyp(2)*sum(max(abs(Z3),[],2)); %                 
        for task=1:M
            val = val + mean((Y_cell{task} - X_cell{task}*Z1*V(:,task)).^2)/2;
        end
    end
end

