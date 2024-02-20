function [Population,Fitness] = EnvironmentalSelection(Fitness,Population,functionvalue,N)
% The environmental selection of CMOEA-MS

%--------------------------------------------------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

   %% Environmental selection
%   ycbcr_msenew = ycbcr_mse<=1&ycbcr_mse>=0.5
%     Next = Fitness < 1 & ycbcr_msenew';
     Next = Fitness < 1;
     index = find(Next);
%      N = size(Population,1)/2
     if sum(Next) < N
        [~,Rank] = sort(Fitness);
        Next(Rank(1:N)) = true;
    elseif sum(Next) > N
        Del  = Truncation(functionvalue(index,:),sum(Next)-N);
        Temp = find(Next);
        Next(Temp(Del)) = false;
    end

%     if sum(Next) < N
%         [~,Rank] = sort(Fitness);
%         Next(Rank(1:N)) = true;
%     elseif sum(Next) > N
%         Del  = Truncation(functionvalue(index,:),sum(Next)-N);
%         Temp = find(Next);
%         Next(Temp(Del)) = false;
%     end
    index = find(Next);
%     appindex =  find(Next==0);
%     Populationnew = Population;
    Population = Population(index,:);
    Fitness    = Fitness(1,index);
%     aim = functionvalue(index,:);
%     append = Populationnew(appindex,:);
%     appaim = functionvalue(appindex,:);
end

function Del = Truncation(PopObj,K)
% Select part of the solutions by truncation

    Distance = pdist2(PopObj,PopObj,'cosine');
    Distance(logical(eye(length(Distance)))) = inf;
    Del = false(1,size(PopObj,1));
    while sum(Del) < K
        Remain = find(~Del);
        Temp = sort(Distance(Remain,Remain),2);
        [~,Rank] = sortrows(Temp);
        Del(Remain(Rank(1))) = true;
    end
end