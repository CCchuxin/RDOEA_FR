% 计算适应度
function fitness = calcFitness(functionvalue, pop, lambda, rateLimits)

popSize = size(pop, 1);
fitness = zeros(popSize, 1);
mse = zeros(popSize, 1);
rate = zeros(popSize, 1);
for i=1:popSize
    % 计算MSE损失函数
    mse(i) = functionvalue(i,2);
    % 计算压缩后的码率
    rate(i) = functionvalue(i,1);
    % 计算适应度：MSE和码率的加权和
     % 判断码率是否超出限制范围
    if rate(i) <= rateLimits(end) && rate(i) >= rateLimits(1)
        % 如果码率在限制范围内，则不进行惩罚
        fitness(i) = lambda*mse(i);
    else
        % 如果码率超出限制范围，则进行惩罚
        if rate(i) < rateLimits(1)
            fitness(i) = lambda*mse(i) + (1-lambda)*abs(rateLimits(1)-rate(i));
        else
            fitness(i) = lambda*mse(i) + (1-lambda)*abs(rateLimits(end)-rate(i));
        end
    end
end
end