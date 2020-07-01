function [cF, cS, cerrV, ciEpch, crunStats] = ONMTF_SCR(X, k, lnModeStr, prams, maxIter, recRate, initMode, dbgArgs, stopVal)
%ONMTF_SCR Run ONMTF_SCR to find k clusters in each graph in X.
%   Description:
%      Uses the ONMTF_SCR implementation from 
%      https://github.com/ZilongBai/KDD2017.git (Bai et al. 2017). 
%      This function uses ONMTF_SCR to find a k clustering for each 
%      symmetric graph in X and its respective cluster relationship 
%      matrix S.
%
%      Factorization implemented in this function:
%       X(i) = F(i)*S(i)*F(i)' where:
%        F(i)'*F(i) = I
%        values of F(i) and S(i) are non-negative
%        clusters in F(i) are spatially cohesive
%
%      Bai, Zilong, et al. "Unsupervised network discovery for brain imaging 
%       data." Proceedings of the 23rd ACM SIGKDD International Conference on 
%       Knowledge Discovery and Data Mining. 2017.
%
%   Input:
%      X - n x n x m multi-graph matrix where each n x n matrix is
%       symmetric
%      k - number of components (clusters) to generate
%      lnModeStr - unused
%      prams - learning parameter for Bai implementation
%         eps - close to 0 value
%      maxIter - maximum number of iterations
%      recRate - unused
%      initMode - unused
%      dbgArgs - unused
%      stopVal - unused
%
%   Output:
%      cF - n x k cluster membership matrix for each matrix in X
%      cS - k x k cluster relationship matrix for each matrix in X
%      cerrV - 1 x maxIter error vectors for each matrix in
%       X
%      ciEpch - final iteration for each matrix in X
%      crunStats - unused
%   
%   Author:
%      Wrapper code to achieve desired functionality: Kendrick Li [5-17-2020]

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
  
  %% Included onmtf solver
  cF = cell(1, g); cS = cell(1, g);
  cerrV = cell(1, g); ciEpch = zeros(1, g);
  crunStats = cell(1, g);
  for iX = 1:g
    snglX = abs(X(:, :, iX));
    
    [cF{iX}, cS{iX}, ~, cerrV{iX}, ~] = ...
      Solver(snglX, 0, k, maxIter, prams.eps, zeros(n));
    ciEpch(iX) = maxIter;
  end
end

% retrived from https://github.com/ZilongBai/KDD2017.git
function [F,M,ERR,Obj,Conti] = Solver(X,beta,k,MAX,epsilon,Theta)
  %
  % Solves both the Unsupervised Node Discovery subproblem and the Edge Discovery subproblem (by Zilong Bai, KDD Lab @ University of California, Davis)
  %
  % The Unsupervised Node Discovery subproblem is solved with multiplicative update rules for F and G. 
  %% The detailed formulation and deductions can be found in our paper Unsupervised Network Discovery for Brain Imaging Data, KDD'17. 
  %%% The deductions invlolve techniques similar to Orthogonal Nonnegative Matrix Factorization (Ding et al. 2006).
  %
  % The Edge Discovery subproblem is solved with nonnegative least squares.
  %
  % Input
  %   X: N x N matrix where N is the number of voxels. 
  %      Absolute Pearson Correlation graph for all pairs of voxels according to their temporal patterns. 
  %      In the paper we evaluated all voxels within the brain region on one slice (36th).
  %   pc: Scalar. Weight of the continuity regularization term, i.e. beta in the paper.
  %   k: Integer. Number of Nodes to be discover
  %   MAX: Integer.  maximum number of iterations
  %   epsilon: Scalar. An extremely small postive scalar to replace non-positive elements to avoid multiplicative update rules being trapped by zero values.
  %   Theta: N x N matrix where N is the number of voxels. Penalty matrix calculated according to the spatial coordinates of the voxels.
  %
  % Output
  %   F: N x k matrix.
  %      Cluster indicator matrix for discovered Nodes. Each Node is inidcated by a column vector of F.
  %   M: k x k matrix.
  %      Inter-/intra-cluster correlation matrix for discovered Edges.
  %   ERR, Obj, Conti: Vectors for debugging purpose. 
  %                    Record the reconstruction error, objective function value, and the continuity regularization term value of each iteration while solving the Unsupervised Node discovery subproblem

  [spatio, spatioy] = size(X); % X is a symmetric absolute Pearson correlation matrix. Hence spatio equals spatioy, denoting the number of vertices on this correlation graph, i.e. the voxels in the spatial domain. Note they are NOT the spatial coordinates of the voxels.

  %Gaussian Kernel Theta for spatial continuity regularization can be pre-calculated.

  %Initialization

  F = rand(spatio, k);

  G = rand(spatioy, k);

  %=== Node Discovery Subproblem

  % Start alternative solving process for F and G. F is the indicator matrix for Nodes.

  ERR = zeros(1, MAX);
  Obj = zeros(1, MAX);
  Conti = zeros(1, MAX);

  for iteration = 1:MAX
  % Updating F with multiplicative update rule

  SQF = ((X*G)./(F*(G'*G)+beta*Theta*F + F*(F'*X*G - (G'*G) -F'*beta*Theta*F))); % Lambda in the paper is substituted with its explicit expression with F, G, pc, and Theta. 
  SQF(find(SQF<=0)) = epsilon; 
  F = F.*(sqrt(SQF));
  %F(find(F<=0)) = epsilon;

  % Updating G with multiplicative update rule
  G = G.*((X'*F)./(G*(F'*F)));
  G(find(G<=0)) = epsilon;

  % Normalizing columns of F
  for col = 1:k
    normF = norm(F(:,col));
          F(:,col) = F(:,col)./normF;
    G(:,col) = G(:,col).*normF;
  end

  ERR(iteration) = norm(X-F*G','fro')/norm(X,'fro');
  Obj(iteration) = (norm(X-F*G','fro'))^2 + trace((F')*Theta*F);
  Conti(iteration) = trace(F'*Theta*F); 
  end

  %=== Edge Discovery Subproblem

  % Solving symmetrical M with Kronecker product and Nonnegative Least Squares. M records the Edges. 

  vecX = reshape(X,spatio*spatioy,1);

  kronF = kron(F,F);

  vecM = lsqnonneg(kronF,vecX);

  M = reshape(vecM,k,k);
end
