function [bestRunRes, runRes] = multiRESCAL(nmRuns, cores, X, errMthd, args, stopVal)
%multiRESCAL Runs RESCAL multiple times and returns all runs and the best run (least error)
%   Input:
%      nmRuns - number of runs to compute
%      cores - number of cores to run runs on
%      X - n x n x m multi-graph matrix where each n x n matrix is
%       symmetric
%      errMthd - set to either of the two options
%         'reconstErr': use only reconstruction error
%         'weightedErr': use a weighted error between orthogonality and 
%          reconstruction error
%      args - the arguments for the method implementation, see RESCAL
%      stopVal - stop value for method implementation, may be unused
%       depending on implementation, see RESCAL
%
%   Output:
%      bestRunRes - least error result
%      runRes - all run results
%   
%   Author:
%      Kendrick Li [5-17-2020]

    if ~exist('stopVal', 'var')
      stopVal = 1.5e-4;
    end
    
    %% do many runs
    runRes = cell(1, nmRuns);
    nmSbj = size(X, 3);
    if args.dbgArgs.plotFlag || cores < 2
        for iRuns = 1:nmRuns
            [runRes{iRuns}.F, runRes{iRuns}.S, ...
             runRes{iRuns}.errV, iEpch, ...
             runRes{iRuns}.runStats] = ...
                RESCAL(X, args.k, args.lnModeStr, args.params, ...
                               args.maxIter, args.recRate, ...
                               args.initMode, args.dbgArgs, stopVal);

            [runRes{iRuns}.fErr, ~] = ...
              computeRepError(X, runRes{iRuns}.F, runRes{iRuns}.S);
            runRes{iRuns}.endEpch = iEpch/args.recRate;
        end
    else
        parfor (iRuns = 1:nmRuns, cores)
            [runRes{iRuns}.F, runRes{iRuns}.S, ...
             runRes{iRuns}.errV, iEpch, ...
             runRes{iRuns}.runStats] = ...
                RESCAL(X, args.k, args.lnModeStr, args.params, ...
                               args.maxIter, args.recRate, ...
                               args.initMode, args.dbgArgs, stopVal);

            [runRes{iRuns}.fErr, ~] = ...
              computeRepError(X, runRes{iRuns}.F, runRes{iRuns}.S);
            runRes{iRuns}.endEpch = iEpch/args.recRate;
        end
    end
    
    %% Extract data and compute score
    runsFinErr = zeros(1, nmRuns);
    for iRuns = 1:nmRuns
        if strcmp(errMthd.method, 'reconstErr')
            % choose one with best reconst err
            runsFinErr(iRuns) = runRes{iRuns}.fErr;
        elseif strcmp(errMthd.method, 'weightedErr')
            % weighted sum of average per sbj reconst err
            %  vs F'F error
            rF = runRes{iRuns}.F;
            runsFinErr(iRuns) = ...
                errMthd.scWgt*(runRes{iRuns}.fErr/nmSbj) + ...
                (1 - errMthd.scWgt)*norm(rF'*rF - eye(args.k), 'fro')^2;
        end
    end
    
    %% Save the best scoring representation
    [~, iMin] = min(runsFinErr);
    bestRunRes = runRes{iMin};
end

