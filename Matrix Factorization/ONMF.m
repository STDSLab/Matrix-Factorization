function [W_Hist, H_Hist, accHist, objHist, timeHist] = ONMF(X, gnd, k, params, W0, H0)
%ONMF Run ONMF to find a WH matrix factorization for X.
%   Description:
%      Original Orthogonal NMF formulation using Lagrange multipliers and multiplicative
%      update rules
%
%      Factorization implemented in this function:
%       X = W*H where:
%        columns of W are orthogonal
%        values of W and H are non-negative
%
%      Ding, Chris, et al. "Orthogonal nonnegative matrix t-factorizations for 
%	    clustering." Proceedings of the 12th ACM SIGKDD international conference 
%       on Knowledge discovery and data mining. 2006.
%
%   Input:
%      X - n x d non-negative matrix
%      gnd - the ground truth vector
%      k - number of components (clusters) to generate
%      params - learning parameters, contains:
%         T - number of iterations
%         eps - small number which represents 0 (to avoid zero-locking)
%         eps2 - small number for division
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
  if isempty(params.eps2), eps2 = eps^2;
    else, eps2 = params.eps2; end
  
  %% init
  W_Hist = cell(1, T); H_Hist = cell(1, T);
  accHist = zeros(1, T); objHist = zeros(1, T);
  timeHist = zeros(1, T);
  
  %k = length(unique(gnd));
  [n, d] = size(X);
  %rndSel = randperm(n);
  %H0 = X(rndSel(1:k), :);
  
  W0 = rand(n,k);
  H0 = rand(k,d);
  nmSvW = 0; nmSvH = 0;
  tic;
  for t = 1:T
    %% constrain
    W0(W0 < eps) = eps;
    %figure(1); imagesc(W0); colorbar;
    %figure(2); imagesc(H0); colorbar;

    %% record results
    [W_Hist, nmSvW] = storeHist(W_Hist, W0, nmSvW, t);
    [H_Hist, nmSvH] = storeHist(H_Hist, H0, nmSvH, t);

    [~,res] = max(W0, [], 2);
    res = bestMap(gnd,res);
    accHist(t) = length(find(gnd == res))/length(gnd);
    objHist(t) = norm(X-W0*H0, 'fro')^2; 
    %obj(t) = norm(X - W0*H0,'fro');

    %% method implementation
    XH= X*H0'; 
    %W0 = W0.*XH./(1e-9+W0*(W0'*XH));
    %H0 = H0.*(W0'*X)./((W0'*W0)*H0);

    H0 = H0.*((W0'*X)./(W0'*W0*H0 + eps2));
    %H0 = H0.*((X'*W0)./(H0'*(W0')*W0 + eps2))';
    W0 = W0.*sqrt((XH)./(W0*W0'*XH + eps2));
    
    %% time
    timeHist(t + 1) = toc;
  end
  W0(W0 < eps) = eps;

  W_Hist{end + 1} = W0;
  H_Hist{end + 1} = H0;
  
  [~, res] = max(W0, [], 2);
  res = bestMap(gnd, res);
  accHist(end + 1) = length(find(gnd == res))/length(gnd);
  objHist(end + 1) = norm(X-W0*H0 ,'fro')^2;
end
