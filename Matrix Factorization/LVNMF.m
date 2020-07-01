function [W_Hist, H_Hist, accHist, objHist, timeHist] = LVNMF(X, gnd, k, params, W0, H0)
%LVNMF Run LVNMF to find a WH matrix factorization for X.
%   Description:
%      Solves the ONMF objective by maximizing the volume between the columns of W
%
%      Factorization implemented in this function:
%       X = W*H where:
%        columns of W are maximally different
%        values of W and H are non-negative
%
%      Liu, Tongliang, Mingming Gong, and Dacheng Tao. "Large-cone nonnegative matrix 
%       factorization." IEEE transactions on neural networks and learning systems 
%       28.9 (2016): 2129-2142.
%
%   Input:
%      X - n x d non-negative matrix
%      gnd - the ground truth vector
%      k - number of components (clusters) to generate
%      params - learning parameters, contains:
%         T - number of iterations
%         eps - small number which represents 0 (to avoid zero-locking)
%         eta - small number for division
%         delta - min change to break out of inner loop
%         lambda - weight of volume penalty in objective
%         stallRatio - if change is below the stallRatio, then detect as stalling
%         dftCntdwn - number of stalls until we exit the loop
%         mxInnerLoop - maximum number of iterations for the inner update loops
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
%      Kendrick Li [10-31-2019]

  if isempty(params.T), T = 200;
    else, T = params.T; end
  if isempty(params.eps), eps = 1e-5;
    else, eps = params.eps; end
  if isempty(params.eta), eta = 0.00075;
    else, eta = params.eta; end
  if isempty(params.delta), delta = 0.001;
    else, delta = params.delta; end
  if isempty(params.lambda), lambda = 5;
    else, lambda = params.lambda; end
  if isempty(params.stallRatio), stallRatio = 0.05;
    else, stallRatio = params.stallRatio; end
  if isempty(params.dftCntdwn), dftCntdwn = 5;
    else, dftCntdwn = params.dftCntdwn; end
  if isempty(params.mxInnerLoop), mxInnerLoop = 50;
    else, mxInnerLoop = params.mxInnerLoop; end
	
  W_Hist = cell(1, T); H_Hist = cell(1, T);
  accHist = zeros(1, T); objHist = zeros(1, T);
  timeHist = zeros(1, T);
  
  [n, d] = size(X);
  
  W0 = rand(n,k);
  H0 = rand(k,d);
  nmSvW = 0; nmSvH = 0;
  tic;
  for t = 1:T
    %% record result
    [W_Hist, nmSvW] = storeHist(W_Hist, W0, nmSvW, t);
    [H_Hist, nmSvH] = storeHist(H_Hist, H0, nmSvH, t);

    [~,res] = max(W0, [], 2);
    res = bestMap(gnd,res);
    accHist(t) = length(find(gnd == res))/length(gnd);
    objHist(t) = norm(X-W0*H0, 'fro')^2;
    
    %% method implementation
    Hprev = H0; pCng = inf; cntdwn = dftCntdwn;
    %cngHist = [];
    innerCnt = 0;
    while innerCnt < mxInnerLoop
      Htmp = Hprev; Htmp(Htmp < eps) = eps;
      Hnxt = Htmp - eta*(W0'*W0*Htmp - W0'*X);
      Hnxt(Hnxt < 0) = 0;
      
      objPrev = loc_obj1(W0, Hprev, X, lambda);
      cng = abs((objPrev - loc_obj1(W0, Hnxt, X, lambda))/objPrev);
      
      if abs(cng - pCng)/pCng <= stallRatio || cntdwn ~= dftCntdwn
        cntdwn = cntdwn - 1;
      end
      
      %cngHist(end + 1) = cng;
      if cng <= delta || cntdwn <= 0
        H0 = Hnxt;
        break;
      end
      Hprev = Hnxt; pCng = cng; innerCnt = innerCnt + 1;
    end
    
    Wprev = W0; pCng = inf; cntdwn = dftCntdwn;
    %cngHist = [];
    innerCnt = 0;
    while innerCnt < mxInnerLoop
      Wtmp = Wprev; Wtmp(Wtmp < eps) = eps;
      Wnxt = Wtmp - eta*(Wtmp*H0*(H0') - X*H0' - lambda*n*(pinv(Wtmp)'));
      Wnxt(Wnxt < 0) = 0;
      Wnxt = normc(Wnxt);
      
      objPrev = loc_obj1(Wprev, H0, X, lambda);
      cng = abs((objPrev - loc_obj1(Wnxt, H0, X, lambda))/objPrev);
      
      if abs(cng - pCng)/pCng <= stallRatio || cntdwn ~= dftCntdwn
        cntdwn = cntdwn - 1;
      end
      
      %cngHist(end + 1) = cng;
      if cng <= delta || cntdwn <= 0
        W0 = Wnxt;
        break;
      end
      Wprev = Wnxt; pCng = cng; innerCnt = innerCnt + 1;
    end
    
    %% time
    timeHist(t) = toc;
  end

  %figure(1); imagesc(H0)
  %figure(2); imagesc(W0)
  %figure(3); plot(objHist)
  
  W_Hist{end + 1} = W0;
  H_Hist{end + 1} = H0;
  
  [~, res] = max(W0, [], 2);
  res = bestMap(gnd, res);
  accHist(end + 1) = length(find(gnd == res))/length(gnd);
  objHist(end + 1) = norm(X-W0*H0 ,'fro')^2;
end

  function v = loc_obj1(W, H, X, lambda)
    n = size(H, 2);
    v = (1/n)*norm(X - W*H)^2 - lambda*log(det(W'*W));
  end