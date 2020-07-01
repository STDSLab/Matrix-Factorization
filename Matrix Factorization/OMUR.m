function [W_Hist, H_Hist, accHist, objHist, timeHist] = OMUR(X, gnd, k, params, W0, H0)
%OMUR Run OMUR to find a WH matrix factorization for X.
%   Description:
%      Solves the ONMF objective projected on the Stiefel manifold to guarantee 
%      orthogonality. the Stiefel manifold is the space of all solutions that 
%      fulfill the orthogonal constraints (set of all orthonormal k-frames in Rn)
%
%      Factorization implemented in this function:
%       X = W*H where:
%        W is projected onto the Stiefel manifold
%        values of W and H are non-negative
%
%      Choi, Seungjin. "Algorithms for orthogonal nonnegative matrix factorization." 
%       2008 ieee international joint conference on neural networks (ieee world 
%       congress on computational intelligence). IEEE, 2008.
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
  
  [n, d] = size(X);
  %rndSel = randperm(n);
  %H0 = X(rndSel(1:k), :);
  
  W0 = rand(n,k);
  %W0 = initW(n, k);
  H0 = rand(k,d);
  nmSvW = 0; nmSvH = 0;
  tic;
  for t = 1:T
    %% constrain
    W0 = normc(W0);
    W0(W0 < eps) = eps;
    
    %figure(1); imagesc(W0); colorbar;
    %figure(2); imagesc(H0); colorbar;
    
    %% record results
    %{
    svFlg = true;
    if nmSvW ~= 0
      tmpCng = norm(W_Hist{nmSvW} - W0, 'fro')/norm(W_Hist{nmSvW}, 'fro');
      if tmpCng < 0.01
        svFlg = false;
      end
    end
    if svFlg
      W_Hist{t} = W0;
      nmSvW = t;
    end
    
    svFlg = true;
    if nmSvH ~= 0
      tmpCng = norm(H_Hist{nmSvH} - H0, 'fro')/norm(H_Hist{nmSvH}, 'fro');
      if tmpCng < 0.01
        svFlg = false;
      end
    end
    if svFlg
      H_Hist{t} = H0;
      nmSvH = t;
    end
    %}
    [W_Hist, nmSvW] = storeHist(W_Hist, W0, nmSvW, t);
    [H_Hist, nmSvH] = storeHist(H_Hist, H0, nmSvH, t);

    [~,res] = max(W0, [], 2);
    res = bestMap(gnd,res);
    accHist(t) = length(find(gnd == res))/length(gnd);
    objHist(t) = norm(X-W0*H0, 'fro')^2;
    
    %% method implementation
    H0 = H0.*((W0'*X)./(W0'*W0*H0 + eps2));
    W0 = W0.*((X*H0')./(W0*H0*X'*W0 + eps2));
    
    %% time
    timeHist(t + 1) = toc;
  end
  W0 = normc(W0);
  W0(W0 < eps) = eps;

  W_Hist{end + 1} = W0;
  H_Hist{end + 1} = H0;
  
  [~, res] = max(W0, [], 2);
  res = bestMap(gnd, res);
  accHist(end + 1) = length(find(gnd == res))/length(gnd);
  objHist(end + 1) = norm(X-W0*H0 ,'fro')^2;
end

function iW = loc_initW(n, k)
  iW = zeros(n, k);
  cl = 1;
  for iV = 1:n
    iW(iV, cl) = 1;
    cl = cl + 1;
    if cl > k
      cl = 1;
    end
  end
  iW = normc(iW);
  Wmix = randperm(n);
  iW = iW(Wmix, :);
end