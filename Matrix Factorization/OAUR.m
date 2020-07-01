function [W_Hist, H_Hist, accHist, objHist, timeHist] = OAUR(X, gnd, k, params, W0, H0)
%OAUR Run OAUR to find a WH matrix factorization for X.
%   Description:
%      Solves a modified ONMF objective with an orthogonality penalty instead of
%      using the Lagrange multipliers. Additionally changes multiplicative updates
%      rules to additive update rules in order to provide a robust convergence guarantee 
%      Zero-locking is addressed by using a small number at zero instead of zero
%
%      Factorization implemented in this function:
%       X = W*H where:
%        orthogonality penalty on the columns of W
%        values of W and H are non-negative
%
%      Mirzal, Andri. "A convergent algorithm for orthogonal nonnegative matrix 
%       factorization." Journal of Computational and Applied Mathematics 260 
%       (2014): 149-166.
%
%   Input:
%      X - n x d non-negative matrix
%      gnd - the ground truth vector
%      k - number of components (clusters) to generate
%      params - learning parameters, contains:
%         T - number of iterations
%         eps - small number which represents 0 (to avoid zero-locking)
%         delta - small number for division
%         alpha - orthogonality component weight (larger values increase
%          orthogonality importance)
%         step - how much delta changes in the update
%      W0 - initial W matrix - unused
%      H0 - initial H matrix - unused
%
%   Output:
%      W_Hist - W matrix history cell array. Contains W for each iteration. If 
%       W did not change much since the last saved W, W is not saved.
%      H_Hist - H matrix history cell array. Contains H for each iteration. If 
%       H did not change much since the last saved H, H is not saved.
%      accHist - clustering accuracy on each iteration
%      objHist - the value for the objective on each iteration
%      timeHist - the time passed since the algorithm started on each iteration
%   
%   Author:
%      Kendrick Li [11-5-2019]

  %% extract params
  if isempty(params.T), T = 200;
    else, T = params.T; end
  if isempty(params.eps), eps = 1e-5;
    else, eps = params.eps; end
  if isempty(params.delta), delta = eps^2;
    else, delta = params.delta; end
  if isempty(params.alpha), alpha = 100;
    else, alpha = params.alpha; end
  if isempty(params.step), step = 10;
    else, step = params.step; end
  
  %% init
  W_Hist = cell(1, T); H_Hist = cell(1, T);
  accHist = zeros(1, T); objHist = zeros(1, T);
  timeHist = zeros(1, T);
  
  % this algorithm constrains the rows of H0 as orthogonal, so we transpose
  % X and save H0 as W0' and W0 as H0';
  X = X';
  [d, n] = size(X);
  %rndSel = randperm(n);
  %H0 = X(rndSel(1:k), :);
  
  W0 = rand(d,k);
  H0 = rand(k,n);
  nmSvW = 0; nmSvH = 0;
  tic;
  for t = 1:T
    %% record results
    [W_Hist, nmSvW] = storeHist(W_Hist, H0', nmSvW, t);
    [H_Hist, nmSvH] = storeHist(H_Hist, W0', nmSvH, t);

    [~,res] = max(H0);
    res = bestMap(gnd,res);
    accHist(t) = length(find(gnd == res))/length(gnd);
    objHist(t) = norm(X-W0*H0, 'fro')^2;
    
    %% method implementation
    %H0 = H0.*((W0'*X)./(W0'*W0*H0));
    %W0 = W0.*((X*H0')./(W0*H0*X'*W0));
    
    gW0 = W0*H0*(H0') - X*H0';
    W0hat = W0; W0hat(gW0 < 0 & W0hat < eps) = eps;
    
    W0 = W0 - ((W0hat.*gW0)./(W0hat*H0*(H0') + delta));
    deltaH0 = delta;
    
    intJ = loc_J(X, W0, H0, alpha);
    gH0 = W0'*W0*H0 - W0'*X + alpha*H0*(H0')*H0 - alpha*H0;
    H0hat = H0; H0hat(gH0 < 0 & H0hat < eps) = eps;
    while true
      tmpH0 = H0 - ((H0hat.*gH0)./(W0'*W0*H0hat + alpha*H0hat*(H0hat')*H0hat + deltaH0));
      deltaH0 = deltaH0*step;
      if loc_J(X, W0, tmpH0, alpha) <= intJ
        H0 = tmpH0;
        break
      end
    end
    
    %% time
    timeHist(t + 1) = toc;
  end

  W_Hist{end + 1} = H0';
  H_Hist{end + 1} = W0';
  
  [~, res] = max(H0);
  res = bestMap(gnd, res);
  accHist(end + 1) = length(find(gnd == res))/length(gnd);
  objHist(end + 1) = norm(X-W0*H0 ,'fro')^2;
end

function obj = loc_J(A, B, C, alpha)
  obj = (1/2)*norm(A - B*C, 'fro')^2 + (alpha/2)*norm(C*C' - eye, 'fro')^2;
end