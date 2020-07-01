function [F, S, errV, iEpch, runStats] = RESCAL(X, k, lnModeStr, prams, maxIter, recRate, initMode, dbgArgs, stopVal)
%RESCAL Run RESCAL to find a group-wise clustering with k clusters for X.
%   Description:
%      Implements the RESCAL algorithm from Nickel et al. 2011 modified
%      slightly. This function uses RESCAL to find a group-wise k 
%      clustering for the symmetric matrices in X and the respective
%      cluster relationship matrix S.
%
%      Factorization implemented in this function:
%       X(i) = F*S(i)*F' with no additional constraints or penalties
%
%      Nickel, Maximilian, Volker Tresp, and Hans-Peter Kriegel. "A 
%       three-way model for collective learning on multi-relational 
%       data." Icml. Vol. 11. 2011.
%
%   Input:
%      X - n x n x m multi-graph matrix where each n x n matrix is
%       symmetric
%      k - number of components (clusters) to generate
%      lnModeStr - unused (RESCAL update rules used to update values)
%      prams - unused
%      maxIter - maximum number of iterations
%      recRate - step size for measuring the error
%      initMode - unused (random non-negative F and S used)
%      dbgArgs - unused
%      stopVal - unused
%
%   Output:
%      F - n x k cluster membership matrix from the iteration with the
%       lowest error
%      S - k x k x m multi-graph cluster relationship matrix from the
%       iteration with the lowest error
%      errV - 1 x round(maxIter/recRate) error vector where error is 
%       measured for each [recRate] iteration, e.g., with a recRate of 5
%       each value is the error for each 5th iteration
%      iEpch - final iteration number
%      runStats - unused
%   
%   Author:
%      Kendrick Li [5-17-2020]

  %% Setup
  %{
  eta = prams.eta; beta = prams.beta;
  eps = prams.eps; compRat = prams.compRat;
  
  if ~exist('stopVal', 'var')
    stopVal = 1.5e-4;
  end
  
  if strcmp(lnModeStr, 'Adam')
    lnMode = 1;
  elseif strcmp(lnModeStr, 'momentum')
    lnMode = 2;
  elseif strcmp(lnModeStr, 'gradient')
    lnMode = 3;
  else
    warning('unknown learning mode, defaulting to Adam');
  end
  
  if exist('initMode', 'var') == 0
    initMode = 'c';
  end
  
  if strcmp(compRat.mode, 'def')
    rat1 = compRat.rat1; rat2 = compRat.rat2;
    rat3 = compRat.rat3; rat4 = compRat.rat4;
  elseif strcmp(compRat.mode, 'cmp')
    rat1 = compRat.rat; rat2 = 1;
    rat3 = 1 - compRat.rat; rat4 = 0;
  elseif strcmp(compRat.mode, 'none')
    rat1 = 1; rat2 = 1;
    rat3 = 1; rat4 = 0;
  end
  %}
  
  [~, n, g] = size(X);
  
  if exist('maxIter', 'var') == 0
    maxIter = 10000;
  end
  if exist('recRate', 'var') == 0
    recRate = max(1, round(maxIter/1000));
  end
  
  if exist('dbgArgs', 'var') == 0
    dbgArgs.flag = false;
    dbgArgs.plotFlag = false;
  end
    
  %F = A;
  %S = R;
  
  %% Initialize
  %[F, ~] = ...
  %  nOF(n, k, 0.01, 0.95, 100000, 1000, ...
  %      initMode);
  F = rand(n, k);
  S = rand(k, k, g);

  %% solver
  %idnMat = eye(k);
  errCurs = 1; bestErr = inf;
  bestF = zeros(size(F)); bestS = zeros(size(S));
  errV = zeros(1, round(maxIter/recRate));
  runStats = {};
  for iEpch = 1:maxIter
    %% compute error
    if mod(iEpch - 1, recRate) == 0
      % estimate S with X and F
      currErr = computeRepError(X, F, S);
      errV(errCurs) = currErr;
      
      if bestErr > currErr
        bestF = F; bestS = S;
        bestErr = currErr;
      end
      errCurs = errCurs + 1;
    end
    
    %% update F
    fstCmp = zeros(size(F));
    sndCmp = zeros(k);
    for iX = 1:g
      Xo = X(:, :, iX);
      So = S(:, :, iX);
      fstCmp = fstCmp + ...
        Xo*F*So' + ...
        Xo'*F*So;
      
      sndCmp = sndCmp + ...
        So*(F')*F*So' + ...
        So'*(F')*F*So;
    end
    %F = fstCmp/(sndCmp + lambda*idnMat);
    F = fstCmp*pinv(sndCmp);

    %% update S
    Z = kron(F, F); F1 = pinv(F'*F);
    for iX = 1:g
      %S(:, :, iX) = ...
      %  (Z'*Z + lambda*idnMat)\...
      %  Z*reshape(X(:, :, iX), [], 1);
      S(:, :, iX) = ...
        reshape(kron(F1, F1)*...
        Z'*reshape(X(:, :, iX), [], 1), k, k);
    end
  end
  
  %% Use best F and generate S after learning
  currErr = computeRepError(X, F, S);
  if bestErr < currErr
    F = bestF; S = bestS;
  end
end