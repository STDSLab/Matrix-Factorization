function [cF, cS, cerrV, ciEpch, crunStats] = fastMtF(X, k, lnModeStr, prams, maxIter, recRate, initMode, dbgArgs, stopVal)
%fastMtF Run FastMtF to find k clusters in each graph in X.
%   Description:
%      Implements the FastMtF algorithm from Wang et al. 2011 modified
%      to optimize for symmetric graphs. This function uses FastMtF to 
%      find a k clustering for each symmetric graph in X and its 
%      respective cluster relationship matrix S.
%
%      Factorization implemented in this function:
%       X(i) = F(i)*S(i)*F(i)' where:
%        Each row in F(i) contains one non-zero value and the values in F(i) 
%         are constrained to be one or zero.
%        S(i) = inv(F(i)'*F(i))*F(i)'*X*F(i)*inv(F(i)'*F(i))
%
%      Wang, Hua, et al. "Fast nonnegative matrix tri-factorization for 
%       large-scale data co-clustering." Twenty-Second International Joint 
%       Conference on Artificial Intelligence. 2011.
%
%   Input:
%      X - n x n x m multi-graph matrix where each n x n matrix is
%       symmetric
%      k - number of components (clusters) to generate
%      lnModeStr - unused (FastMtF update rules used to update values)
%      prams - unused
%      maxIter - maximum number of iterations
%      recRate - step size for measuring the error
%      initMode - F initialization mode, see nOF
%      dbgArgs - unused
%      stopVal - unused
%
%   Output:
%      cF - n x k cluster membership matrix for each matrix in X
%      cS - k x k cluster relationship matrix for each matrix in X
%      cerrV - 1 x round(maxIter/recRate) error vectors for each matrix in
%       X
%      ciEpch - final iteration for each matrix in X
%      crunStats - unused
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
  %}
  
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
    
  %{
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
  
  cF = cell(1, g); cS = cell(1, g);
  cerrV = cell(1, g); ciEpch = zeros(1, g);
  crunStats = cell(1, g);
  for iX = 1:g
    snglX = X(:, :, iX);
    
    %% Initialize
    [F, ~] = ...
      nOF(n, k, 0.01, 0.95, 100000, 1000, ...
          initMode);
    F(F > 0) = 1; % set limits on vals

    %% fast MtF solver
    %idnMat = eye(k);  cntDwn = -1;
    errCurs = 1; bestErr = inf; 
    bestF = zeros(size(F));
    
    errV = zeros(1, round(maxIter/recRate));
    for iEpch = 1:maxIter
      S = (F'*F)\F'*snglX*F/(F'*F);

      %% compute error
      if mod(iEpch - 1, recRate) == 0
        % estimate S with X and F
        recX = F*S*F';
        currErr = norm(snglX - recX, 'fro')^2;
        
        errV(errCurs) = currErr;

        if bestErr > currErr
          bestF = F;
          bestErr = currErr;
        end
        errCurs = errCurs + 1;
      end

      %{
       temp = inv(G'*G)
       S[:,:] = temp*G'*X*G*temp

        # step 2 compute G
        F_tilde[:,:] = G*S

        G[:,:] = zeros(n, c)

        @time begin
          sum_X[:,:] = sum(X.^2, dims=1)
          sum_F_tilde[:,:] = sum(F_tilde.^2, dims=1)
          sum_result[:,:] = transpose!(sum_X_trans, sum_X) .+ sum_F_tilde
          BLAS.gemm!('T', 'N', -2.0, X, F_tilde, 1.0, sum_result)

          min_res = findmin(sum_result, dims=2)[2]
          for i=1:size(min_res, 1)
            G[i, min_res[i][2]] = 1
          end
    end
        check_membership(G)
        %}
      
      %% update factors
      %{
      % G
      FS = F*S;
      for iN = 1:n
        [~, G()] = ...
          min(sum((FS - snglX(:, iN)).^2));
      end
      %}
      
      % F
      SF = S*F'; F(:) = 0;
      for iN = randperm(n)
        [~, minIdx] = ...
          min(sum((SF - snglX(iN, :)).^2, 2));
        F(iN, minIdx) = 1;
      end
      
      for iK = 1:k
        memNum = sum(F);
        if (memNum(iK) == 0)
          [~, mxIdx] = max(memNum);
          nonZ = find(F(:, mxIdx) > 0);
          nonZ = nonZ(randperm(length(nonZ)));
          F(nonZ(1), :) = 0;
          F(nonZ(1), iK) = 1;
        end
      end
    end

    %% Use best F and generate S after learning
    S = (F'*F)\F'*snglX*F/(F'*F);
    recX = F*S*F';
    currErr = norm(snglX - recX, 'fro')^2;
    if bestErr < currErr
      F = bestF;
      S = (F'*F)\F'*snglX*F/(F'*F);
    end
    
    cF{iX} = F;
    cS{iX} = S;
    cerrV{iX} = errV;
    ciEpch(iX) = iEpch;
  end
end