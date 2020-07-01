function [F, err] = nOF(n, k, eta, beta, maxIter, recRate, mode)
%nOF Random F matrix generator
%   Description:
%      Generates a random cluster membership matrix F from one of several 
%      different methods.
%
%   Input:
%      n - number of rows (objects)
%      k - number of columns (clusters)
%      eta - learning rate (only used for 'gd' mode)
%      beta - beta parameter for momentum (only used in 'gd' mode)
%      maxIter - maximum number of iterations (only used in 'gd' mode)
%      recRate - step size for measuring the error (only used in 'gd' mode)
%      mode - set to following options
%         'gd': constrained gradient descent solver to find an column-wise 
%          orthogonal matrix with non-negative values
%         'r': random matrix with values from 0 to 1
%         'nr': random matrix with values from 0 to 1. Column values are
%          normalized
%         'c': generate a matrix assuming a random clustering of rows such
%          that each row has only one non-zero value corresponding to the
%          cluster (column) it is a member of. Column values are
%          normalized. Each column is guaranteed to have a minimum of one
%          member
%
%   Output:
%      F - n x k cluster membership matrix
%      err - 1 x round(maxIter/recRate) error vector where error is 
%       measured for each [recRate] iteration, e.g., with a recRate of 5
%       each value is the error for each 5th iteration (only used for 'gd'
%       mode)
%   
%   Author:
%      Kendrick Li [5-17-2020]

    %% init
    F = rand(n, k); 
    
    if strcmp(mode, 'gd')
        deltaF = zeros(n, k);
        lambda = rand(n, k); deltaLambda = zeros(n, k);
        err = zeros(1, maxIter/recRate);

        %% gradient descent solver satisfy F'F where F is non-negative
        for iIter = 1:maxIter 
            if mod(iIter, recRate) == 0
                err(int32(iIter/recRate)) = sum(sum(F'*F - eye(k)));
            end
            gLmda = -F;
            giF = F*(F')*F - F - lambda;
            deltaLambda = beta.*deltaLambda + (1 - beta).*(eta.*gLmda);
            deltaF = beta.*deltaF + (1 - beta).*(eta.*giF);
            lambda = lambda + deltaLambda;
            lambda(lambda < 0) = 0;
            F = F - deltaF;
        end
    elseif strcmp(mode, 'r')
        err = 0;
    elseif strcmp(mode, 'nr')
        F = normc(F);
        err = 0;
    elseif strcmp(mode, 'c')
        F = zeros(n, k);
        objMem = [1:k ...
                  randi(k, 1, n - k)];
        objMem = objMem(randperm(n));
        
        for iObj = 1:n
            F(iObj, objMem(iObj)) = 1;
        end
        F = normc(F);
        err = 0;
    end
end

