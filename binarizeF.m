function [binF, clustAssign, mx] = binarizeF(F)
%binarizeF Binarizes an n x k F matrix to produce cluster assignments
%   Description:
%      This function interprets the input n x k matrix as a cluster membership
%      matrix for n objects and k clusters. It computes the maximum value for
%      each row, then sets the column with the maximum value to 1 and all
%      other columns to 0.
%
%   Input:
%      F - n x k matrix containing row-wise membership values from 0 to 1
%
%   Output:
%      binF - the binary F matrix
%      clustAssign - cluster assignment vector 
%      mx - max value for each row
%   
%   Author:
%      Kendrick Li [5-17-2020]

    binF = zeros(size(F));
    [mx, clustAssign] = max(F, [], 2);
    for iRow = 1:length(clustAssign)
        binF(iRow, clustAssign(iRow)) = 1;
    end
end

