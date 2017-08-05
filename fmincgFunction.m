function [all_w, J_history] = fmincgFunction(X, y, num_labels, lambda, num_iters)
% Trains multiple logistic regression classifiers and returns all
% the classifiers in a matrix all_w, where the i-th row of all_w 
% corresponds to the classifier for label i,
% fmincg works similarly to fminunc, but is more efficient when we
% are dealing with large number of parameters.

% variables
n = size(X, 2);

% return variables
all_w = zeros(num_labels, n + 1);

J_history = zeros(num_iters, num_labels);

options = optimset('GradObj', 'on', 'MaxIter', num_iters);
for c = 1:num_labels
    fprintf('\nTrainning k: %f\n', c);
    initial_w = all_w(c, :)';
    [all_w(c,:), J] = fmincg (@(t)(regCostFunction(t, X, (y == c), lambda)), initial_w, options);
    % To capture the cost history on each iteration if the optimization
    % function fmincg don't can not optimze any more complete the history
    % cost with the last value cost evaluated, else record the complete
    % history for the iteration.
    if (length(J)<num_iters)
        aux = zeros(num_iters - length(J),1);
        aux(:) = J(length(J));
        J = [J;aux];
        J_history(:, c) = J;
    else
        J_history(:, c) = J;
    end
end

end
