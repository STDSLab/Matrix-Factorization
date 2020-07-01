function [matHist, svIdx] = storeHist(matHist, currentMat, svIdx, currentIdx, cngPerc)
%storeHist Determine if matrix changed enough to warrent saving in history
%   Description:
%      Saves the current matrix into the matrix history cell array if the matrix
%      is sufficiently different from the last saved matrix.
%
%   Input:
%      matHist - the history cell array
%      currentMat - the current matrix
%      svIdx - the last index which we saved a matrix to
%      currentIdx - the current index in the history cell array
%      cngPerc - save if the average value difference is greater than this percent
%
%   Output:
%      matHist - the modified history cell array
%      svIdx - the updated last index which we saved a matrix to
%   
%   Author:
%      Kendrick Li [11-7-2019]

  if exist('cngPerc', 'var') == 0
    cngPerc = 0.05;
  end

  svFlg = true;
  if svIdx ~= 0
    cng = norm(matHist{svIdx} - currentMat, 'fro') ...
         /norm(matHist{svIdx}, 'fro');
    if cng < cngPerc
      svFlg = false;
    end
  end
  
  if svFlg
    matHist{currentIdx} = currentMat;
    svIdx = currentIdx;
  end
end

