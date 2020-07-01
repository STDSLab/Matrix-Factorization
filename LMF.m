function [F, S, errV, iEpch, runStats] = LMF(X, k, lnModeStr, prams, maxIter, recRate, initMode, dbgArgs, stopVal)
%LMF Run LMF to find a group-wise clustering with k clusters for X.
%   Description:
%      Implements the LMF algorithm from Tang et al. 2009 modified 
%      slightly. This function uses LMF to find a group-wise k 
%      clustering for the symmetric matrices in X and the respective 
%      cluster relationship matrix S.
%
%      Factorization implemented in this function:
%       X(i) = F*S(i)*F' with no additional constraints or penalties
%
%      Tang, Wei, Zhengdong Lu, and Inderjit S. Dhillon. "Clustering with 
%       multiple graphs." 2009 Ninth IEEE International Conference on 
%       Data Mining. IEEE, 2009.
%
%   Input:
%      X - n x n x m multi-graph matrix where each n x n matrix is
%       symmetric
%      k - number of components (clusters) to generate
%      lnModeStr - which optimization to use
%         'Adam': use Adam optimization
%         'gradient': use no optimization (standard gradient descent)
%      prams - learning parameters with four components
%         eta - learning rate
%         beta - beta parameter vector for Adam and momentum (momentum only
%          uses the first value in vector)
%         eps - close to 0 value
%         compRat - unused
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
  eta = prams.eta; beta = prams.beta;
  eps = prams.eps; compRat = prams.compRat;
  
  %{
  if ~exist('stopVal', 'var')
    stopVal = 1.5e-4;
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
  
  %F = P;
  %S = A;
  
  if strcmp(lnModeStr, 'Adam')
    lnMode = 1;
  elseif strcmp(lnModeStr, 'gradient')
    lnMode = 3;
  else
    warning('unknown learning mode, defaulting to Adam');
    lnMode = 1;
  end
  
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
    
  %% Initialize
  % if debug is on, set F as the initial F but with minor noise
  %[P, ~] = ...
  %  nOF(n, k, 0.01, 0.95, 100000, 1000, ...
  %      initMode);
  F = rand(n, k);
  S = rand(k, k, g);

  % set limits on vals
  deltaF = zeros(n, k);
  deltaS = zeros(k, k, g);

  if lnMode == 1
    mF = zeros(n, k); vF = zeros(n, k);
    mS = zeros(k, k, g); vS = zeros(k, k, g);
  end

  %% gradient descent solver
  %idnMat = eye(k); 
  bestErr = inf; errCurs = 1;
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

    %% compute gradients
    gF = zeros(size(F));
    for iX = 1:g
      So = S(:, :, iX);
      gF = gF + ...
        (X(:, :, iX) - F*So*F')*F*So;
    end
    gF = -2*gF;
    
    gS = zeros(size(S));
    for iX = 1:g
      gS(:, :, iX) = ...
        -(F'*(X(:, :, iX)-F*S(:, :, iX)*F')*F);
    end
    
    %% update variables
    if lnMode == 1
      %% Adam
      % F
      betaP = beta.^iEpch;
      mF = beta(1)*mF + (1 - beta(1))*gF;
      vF = beta(2)*vF + (1 - beta(2))*gF.^2;
      etaHat = eta* ...
        (sqrt(1 - betaP(2))/(1 - betaP(1)));
      deltaF = etaHat*mF./(sqrt(vF) + eps);
      
      % S
      mS = beta(1)*mS + (1 - beta(1))*gS;
      vS = beta(2)*vS + (1 - beta(2))*gS.^2;
      etaHat = eta* ...
        (sqrt(1 - betaP(2))/(1 - betaP(1)));
      deltaS = etaHat*mS./(sqrt(vS) + eps);

    elseif lnMode == 3
      %% gradient
      deltaF = eta.*gF;
      deltaS = eta.*gS;
    end
    F = F - deltaF;
    S = S - deltaS;
  end
  
  %% Use best F and generate S after learning
  currErr = computeRepError(X, F, S);
  if bestErr < currErr
    F = bestF; S = bestS;
  end
end