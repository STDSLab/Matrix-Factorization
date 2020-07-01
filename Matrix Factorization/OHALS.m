function [W_Hist, H_Hist, accHist, objHist, timeHist] = OHALS(X, gnd, k, params, W0, H0)
%OHALS Run OHALS to find a WH matrix factorization for X.
%   Description:
%      Solves the ONMF objective through a Alternating Least Squares solution
%      on the columns of W.
%
%      Factorization implemented in this function:
%       X = W*H where:
%        columns of W are orthogonal
%        values of W and H are non-negative
%        hierarchical ALS solution
%
%      Kimura, Keigo, Yuzuru Tanaka, and Mineichi Kudo. "A fast hierarchical 
%       alternating least squares algorithm for orthogonal nonnegative matrix 
%       factorization." Asian Conference on Machine Learning. 2015.
%
%   Input:
%      X - n x d non-negative matrix
%      gnd - the ground truth vector
%      k - number of components (clusters) to generate
%      params - learning parameters, contains:
%         T - number of iterations
%         eps - small number which represents 0 (to avoid zero-locking)
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
  
  %% init
  W_Hist = cell(1, T);
  H_Hist = cell(1, T);
  accHist = zeros(1, T);
  objHist = zeros(1, T);
  timeHist = zeros(1, T);
  
  [n, d] = size(X);
  
  W0 = rand(n,k);
  H0 = rand(k,d);
  nmSvW = 0; nmSvH = 0;
  U = sum(W0,2);
  tic;
  for t = 1:T
    %% constrain
    W0(W0 < eps) = eps;
    H0(H0 < eps) = eps;
    
    %% record result
    [W_Hist, nmSvW] = storeHist(W_Hist, W0, nmSvW, t);
    [H_Hist, nmSvH] = storeHist(H_Hist, H0, nmSvH, t);

    [~,res] = max(W0, [], 2);
    res = bestMap(gnd,res);
    accHist(t) = length(find(gnd == res))/length(gnd);
    objHist(t) = norm(X-W0*H0, 'fro')^2;
  
    %% method implementation
    A = X*H0';
    B = H0*H0';
    
    for j = 1:k
        W0_j = U - W0(:,j);
        h = A(:,j) - W0*B(:,j) + B(j,j)*W0(:,j);
        tW0 = (h - ((W0_j'*h)/(W0_j'*W0_j))*W0_j);
        tW0(tW0 < eps) = eps;
        W0(:,j) = tW0;
        
        W0(:,j) = W0(:,j)./norm(W0(:,j)); % normalize W
        
        U = W0_j + W0(:,j);
    end
    
    C = X'*W0;
    D = W0'*W0;
    
    for j=1:k
        tH0 = C(:,j) - H0'*D(:,j) + D(j,j)*H0(j,:)';
        tH0(tH0 < eps) = eps;
        H0(j,:) = tH0;
    end
    
    %% time
    timeHist(t + 1) = toc;
  end
  W0(W0 < eps) = eps;
  H0(H0 < eps) = eps;

  W_Hist{end + 1} = W0;
  H_Hist{end + 1} = H0;
  
  [~, res] = max(W0, [], 2);
  res = bestMap(gnd, res);
  accHist(end + 1) = length(find(gnd == res))/length(gnd);
  objHist(end + 1) = norm(X-W0*H0 ,'fro')^2;
end