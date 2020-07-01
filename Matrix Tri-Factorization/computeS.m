function S = computeS(X, F)
%computeS Compute the S representation for X given a F matrix assuming that F is column orthogonal
%   Description:
%      This function computes a multi-graph cluster relationship matrix S for
%      a multi-graph symmetric matrix X using a cluster membership matrix F 
%      with n objects, m graphs, and k clusters. This is done by solving for S 
%      from the tri-factorization definition.
%       X = FSF'
%       F'XF = F'FSF'F
%       inv(F'F)F'XFinv(F'F) = S
%
%   Input:
%      X - n x n x m multi-graph matrix where each n x n matrix is
%          symmetric
%      F - n x k cluster membership matrix
%
%   Output:
%      S - k x k x m multi-graph cluster relationship matrix
%   
%   Author:
%      Kendrick Li [5-17-2020]

    g = size(X, 3);
    k = size(F, 2);

    %X = FSF'
    %F'XF = F'FSF'F
    %inv(F'F)F'XFinv(F'F) = S
    
    S = zeros(k, k, g);
    for iX = 1:g
        S(:, :, iX) = (F'*F)\F'*X(:, :, iX)*F/(F'*F);
    end
end

