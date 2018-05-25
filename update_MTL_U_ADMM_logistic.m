function [ U_opt U_history ] = update_MTL_U_ADMM_logistic(U_init, V,X_cell,Y_cell,hyp,opts);

% Last modified on May 25, 2018

%% initialization
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
opts_minfunc.display='none';
opts_minfunc.Method='lbfgs';
opts_minfunc.maxFunEvals = 100;
opts_minfunc.optTol=opts.tol_ADMM;

while iter<=opts.max_iter_ADMM
%     disp(iter)
    % update U
    U = (Z1+Z2+Z3 - L1-L2-L3)/3;
    
    % update auxiliary variables Z
    Z1_old = Z1;
    Z2_old= Z2;
    Z3_old = Z3;
    
    fun_Z1_vec = @(Z1_vector)obj_Z1(Z1_vector,U,L1);
    Z1_vec = minFunc(fun_Z1_vec, Z1(:), opts_minfunc);
    
    Z1 = reshape(Z1_vec, D,K);
    Z2 = prox_L11norm(U + L2, hyp(1)/rho);
    Z3 = prox_L1infnorm(U + L3, hyp(2)/rho);
    
    
    % update Lagrangian variables L
    
    L1 = L1+ U - Z1;
    L2 = L2 + U - Z2;
    L3 = L3 + U - Z3;
    
    % history   
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
            val = val + norm(Y_cell{task} - X_cell{task} * Z1 * V(:,task),2)^2/(2*length(Y_cell{task}));
        end
    end

    function [f, g_vec] = obj_Z1 (Z1_vec,U, L1)
        Z1_mat = reshape(Z1_vec, D, K);
        f = (rho/2) * norm (Z1_mat - U -L1,'fro')^2;
        
        g_mat =rho * (Z1_mat - U - L1);        
        for task=1:M
            [f_temp,g_temp,~] = Loss_logistic(X_cell{task},Y_cell{task}, Z1_mat*V(:,task));
            f =f + f_temp;
            g_mat = g_mat+ g_temp * V(:,task)';
        end
        g_vec = g_mat(:);
    end

end

