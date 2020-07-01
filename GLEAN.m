function [F, S, errV, iEpch, runStats] = GLEAN(X, k, lnModeStr, prams, maxIter, recRate, initMode, dbgArgs, stopVal)
%GLEAN Run GLEAN to find a group-wise clustering with k clusters for X.
%   Description:
%      Implements the GLEAN algorithm from Group Convex Orthogonal 
%      Non-negative Matrix Tri-Factorization with Applications in FC 
%      Fingerprinting. This function uses GLEAN to find a group-wise k
%      clustering for the symmetric matrices in X and the respective cluster
%      relationship matrix S.
%
%      Factorization implemented in this function:
%       X(i) = F*S(i)*F' where:
%        F'*F = I
%        values of F are non-negative
%        S(i) = F'*X*F
%
%      Li, Kendrick T. Group Convex Orthogonal Non-negative Matrix 
%       Tri-Factorization with Applications in FC Fingerprinting. Diss. 
%       University of Cincinnati, 2020.
%
%   Input:
%      X - n x n x m multi-graph matrix where each n x n matrix is
%       symmetric
%      k - number of components (clusters) to generate
%      lnModeStr - which optimization to use
%         'Adam': use Adam optimization
%         'momentum': use momentum optimization
%         'gradient': use no optimization (standard gradient descent)
%      prams - learning parameters with four components
%         eta - learning rate
%         beta - beta parameter vector for Adam and momentum (momentum only
%          uses the first value in vector)
%         eps - close to 0 value
%         compRat - ratio parameters to weigh reconstruction error or
%          orthogonality error, contains 5 components
%            mode - what mode to perform adjustments
%               'def': use all values
%               'cmp': use a ratio
%                (alpha*rec error + (1 - alpha)*orth error)
%               'none': use no ratio
%            rat1 - how much to weigh the reconstruction error (i.e. alpha 
%             when mode is set to 'cmp'. If 'cmp', must be between 0 and 1)
%            rat2 - exponential weighing of matrix element-wise
%             reconstruction error (set to 1 for no weighing)
%            rat3 - how much to weigh the orthogonality error (is set to
%             (1 - rat1) when mode is set to 'cmp' and input value is
%             ignored
%            rat4 - sparcity weight (set to 0 for no sparcity)
%      maxIter - maximum number of iterations
%      recRate - step size for measuring the error
%      initMode - F initialization mode, see nOF
%      dbgArgs - debug arguments
%         flag - set to true if you wish to initialize F = dbgArgs.F with
%          some random noise whose magnitude is set by dbgArgs.noise
%         errARIFlag - set to true if you wish to compute error not by
%          reconstruction error but against a ground truth dbgArgs.F matrix
%         plotFlag - set to true to observe learning plots
%      stopVal - stop value for method implementation, is unused
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
%      runStats - iteration statistics
%         lbls - name for each column in stats
%         stats - stat values
%         converge - true if converged (always false if stopVal is unused)
%   
%   Author:
%      Kendrick Li [5-17-2020]

  %% Setup
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
    lnMode = 1;
  end
  
  [~, n, g] = size(X);
  
  if exist('maxIter', 'var') == 0
    maxIter = 10000;
  end
  if exist('recRate', 'var') == 0
    recRate = max(1, round(maxIter/1000));
  end
  if exist('initMode', 'var') == 0
    initMode = 'c';
  end
  if exist('dbgArgs', 'var') == 0
    dbgArgs.flag = false;
    dbgArgs.plotFlag = false;
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
  
  %% Initialize
  % if debug is on, set F as the initial F but with minor noise
  if dbgArgs.flag
    F = dbgArgs.F + ...
      dbgArgs.noise*(rand(n, k) - 0.5);
  else
    [F, ~] = ...
      nOF(n, k, 0.01, 0.95, 100000, 1000, ...
          initMode);
  end
  dbgErr = [];
  randDispSbj = randi(g);

  % set limits on vals
  F(F < 0) = 0; F(F > 1) = 1;
  deltaF = zeros(n, k);

  if lnMode == 1
    m = zeros(n, k); v = zeros(n, k);
  end

  %% gradient descent solver
  idnMat = eye(k); errCurs = 1; cntDwn = -1;
  bestErr = inf; bestF = zeros(size(F));
  errV = zeros(2, round(maxIter/recRate));
  
  converge = false;
  mxFDiff = zeros(1, maxIter);
  avgFDiff = zeros(1, maxIter);
  vsTruARI_v = zeros(1, maxIter);
  for iEpch = 1:maxIter
    oldF = F;
    
    %% accuracy against a true F vector
    if dbgArgs.errARIFlag
      [~, oldFmem] = binarizeF(oldF);
      [~, truFmem] = binarizeF(dbgArgs.F);
      vsTruARI_v(iEpch) = adjRandIndex(truFmem, oldFmem); % slow
    end
    
    %% compute error
    if mod(iEpch - 1, recRate) == 0
      % estimate S with X and F
      recErr = computeRepError(X, F, computeS(X, F));
      orthErr = norm(F'*F - idnMat, 'fro')^2;
      currErr = rat1*recErr + rat3*orthErr;
      errV(1, errCurs) = recErr;
      errV(2, errCurs) = currErr;
      
      if bestErr > currErr
        bestF = F;
        bestErr = currErr;
      end
      errCurs = errCurs + 1;
    end
    
    %% compute nu
    nu = rat3*((F')*F - idnMat);

    % remove floating point zero noise
    nu(nu < eps) = 0;

    %% compute gradients
    totGF = zeros(n, k);
    U = F*F';
    for iX = 1:g
      Xo = X(:, :, iX);
           
      %{
      U2 = U*U;
      cmp1 = (2/g)*(Xo'*U*Xo + Xo*U*Xo');
      cmp2 = ...
        (1/g)*(U*Xo*U2*Xo' + Xo'*U2*Xo*U + ...
               U*Xo'*U2*Xo + Xo*U2*Xo'*U);
             
      gU = cmp2 - cmp1;
      currGd = 1*gU*F/(F'*F);
      %}
      
      U3 = U*Xo; U2 = Xo*U*U3;
      currGd = (U*U2*F + U2*U*F - ...
          2*Xo*U3*F)/g;
      
      %{
      figure(20); imagesc(testGF); colorbar;
      figure(21); imagesc(currGd); colorbar;
      figure(22); imagesc(currGd - testGF);
      figure(23); imagesc(Xo - F*(F'*Xo*F)*F');
      figure(24); imagesc(U);
      %}

      if rat2 ~= 1
        currGdNg = currGd < 0;
        currGd = abs(currGd).^rat2;
        currGd(currGdNg) = -currGd(currGdNg);
      end
      
      % strip away extremely small gradients
      % caused by floating point error
      % change nmDigits to eps value
      %totGF = totGF + round(currGd, 10);
      totGF = totGF + currGd; % good enough
    end
    errComp = rat1*totGF;
    
    % allow for orthogonal movement on F by 
    %  non-zero F
    nF = F; nF(nF < eps) = eps;
    orthComp = rat3*nF*nu;
    
    gF = errComp + orthComp + rat4; % optional sparsity

    %% update F
    if lnMode == 1
      %% Adam
      betaP = beta.^iEpch;
      m = beta(1)*m + (1 - beta(1))*gF;
      v = beta(2)*v + (1 - beta(2))*gF.^2;
      etaHat = eta* ...
        (sqrt(1 - betaP(2))/(1 - betaP(1)));
      deltaF = etaHat*m./(sqrt(v) + eps);
      %mHat = m/(1 - betaP(1));
      %vHat = v/(1 - betaP(2));
      %deltaF = eta*mHat./(sqrt(vHat) + eps);
    elseif lnMode == 2
      %% momentum
      %deltaNu = beta.*deltaNu + (1 - beta).*((eta*2).*gNu);
      deltaF = beta(1).*deltaF + (1 - beta(1)).*(eta.*gF);
    elseif lnMode == 3
      %% gradient
      %deltaNu = beta.*deltaNu + (1 - beta).*((eta*2).*gNu);
      deltaF = eta.*gF;
    end
    F = F - deltaF;

    %% constrain F
    % between 0 and 1
    F(F < 0) = 0; F(F > 1) = 1;
    
    %% dbg plot error
    if dbgArgs.plotFlag || iEpch == 20001
      dbgArgs.plotFlag = true;
      S = computeS(X, F);
      
      dbgErr(end + 1) = ...
        computeRepError(X, F, S);
      if length(dbgErr) > 100
        dbgErr(1) = [];
      end

      figure(1);
      subplot(2, 5, 1); imagesc(errComp); colorbar; title('Error Component');
      subplot(2, 5, 2); imagesc(orthComp); colorbar; title('Orth Component');
      subplot(2, 5, 3); imagesc(deltaF); colorbar; title('delta F');
      subplot(2, 5, 4); imagesc(F); colorbar; title('F');
      %subplot(2, 5, 5); imagesc(deltaNu); colorbar; title('delta Nu');

      subplot(2, 5, 6); imagesc(X(:, :, randDispSbj)); colorbar; title('X');
      subplot(2, 5, 7); imagesc(F*S(:, :, randDispSbj)*F'); colorbar; title('estX');
      subplot(2, 5, 8); imagesc(S(:, :, randDispSbj)); colorbar; title('S');
      subplot(2, 5, 9); plot(dbgErr); title(num2str(iEpch)); ytickformat('%.1f');
      subplot(2, 5, 10); imagesc(nu); colorbar; title('nu');
    end
    
    %% early stop using max movement of F
    mxFDiff(iEpch) = ...
      max(abs(oldF - F), [], 'all');
    avgFDiff(iEpch) = ...
      norm(oldF - F, 'fro')/(n*k)/eta;
    %if avgFDiff(iEpch) < stopVal
      %% give some grace period 
      %  set larger than 0 to give some 
      %   number of iterations for it to
      %   maybe move fast again
    %  cntDwn = 0;
    %end
    
    %if cntDwn == 0
      %% check to see if this is the best
    %  recErr = computeRepError(X, F, computeS(X, F));
    %  orthErr = norm(F'*F - idnMat, 'fro')^2;
    %  currErr = rat1*recErr + rat3*orthErr;
    %  if bestErr > currErr
    %    bestF = F;
    %  end
    %  converge = true;
    %  break;
    %elseif cntDwn ~= -1
      %% decrement count
    %  cntDwn = cntDwn - 1;
    %end
  end

  runStats.lbls = {'ARI acc', 'F mx diff', 'F avg diff'};
  runStats.stats = [vsTruARI_v; ...
                       mxFDiff; ...
                       avgFDiff];
  runStats.converge = converge;
  
  
  %% Use best F and generate S after learning
  recErr = computeRepError(X, F, computeS(X, F));
  orthErr = norm(F'*F - idnMat, 'fro')^2;
  currErr = rat1*recErr + rat3*orthErr;
  
  if bestErr < currErr
    F = bestF;
  end
  S = computeS(X, F);
end