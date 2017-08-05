function [all_w] = quasiNewton(X, y, num_labels, lambda, num_iters)
% implementation of quasiNewton 2 order optimization
% variables
n = size(X, 2);

% Variable to return 
all_w = zeros(num_labels, n + 1);
J_history = zeros(num_iters, num_labels);

%options = optimset('GradObj', 'on', 'MaxIter', num_iters, 'Display','iter', 'PlotFcns', @optimplotfval);
options = optimset('GradObj', 'on', 'MaxIter', num_iters, 'Display','iter');
for c = 1:num_labels
    fprintf('\nTrainning k: %f\n', c);
    initial_w = all_w(c, :)';
    [all_w(c,:)] = fminunc (@(t)(regCostFunction(t, X, (y == c), lambda)), initial_w, options);
end

end
