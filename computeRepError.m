function [errVal, estX] = computeRepError(X, F, S)
%computeRepError Computes the sum of sum squared error for all matrices in X using the input F and S representation
%   Description:
%      Uses a reshape to achieve fast computation of the error 
%      sum_i(||X_i - F*S_i*F'||^2_Fro) between a multi-graph symmetric matrix
%      X and a matrix factorization consisting of a cluster membership matrix
%      F and multi-graph cluster relationship matrix S with n objects, m 
%      graphs, and k clusters.
%
%   Input:
%      X - n x n x m multi-graph matrix where each n x n matrix is
%       symmetric
%      F - n x k cluster membership matrix
%      S - k x k x m multi-graph cluster relationship matrix
%
%   Output:
%      errVal - sum squared error value
%      estX - X approximation computed from F and S
%   
%   Author:
%      Kendrick Li [5-17-2020]

    [~, nmObj, nmX] = size(X); %nmX = size(X, 3); 
    k = size(F, 2);
    
    rX = reshape(permute(X, [1 3 2]), [nmObj*nmX, nmObj]);
    rS = reshape(S, [k nmX*k]);
    
    fullRep = reshape(permute(reshape(F*rS, [nmObj k nmX]), [1 3 2]), ...
        [nmX*nmObj k])*F';
    estX = permute(reshape(fullRep, [nmObj nmX nmObj]), [1 3 2]);
    errVal = sum(sum((rX - fullRep).^2)); %arrErr
    
    %itErr = 0;
    %estX = zeros(size(X));
    %for iX = 1:nmX
    %    estX(:, :, iX) = F*S(:, :, iX)*F';
    %    itErr = itErr + sum(sum((X(:, :, iX) - estX(:, :, iX)).^2));
    %end
    
    %errVal = arrErr - itErr; % test if identical
end

