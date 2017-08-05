function plotData(X, y, a)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
hold on;

% Find Indices of Positive and Negative Examples
setosa = find(y==1); versicolour = find(y == 2); virginica = find(y == 3);
% Plot Examples
if (a==1)||(a==0)
    plot(X(setosa, 1), X(setosa, 2),'k^', 'MarkerFaceColor', 'b', ...
    'MarkerSize', 7);
else
    plot(X(setosa, 1), X(setosa, 2),'k^', 'MarkerFaceColor', [0.5 0.5 0.5], ...
    'MarkerSize', 7);
end
if (a==2)||(a==0)
    plot(X(versicolour, 1), X(versicolour, 2), 'kd', 'MarkerFaceColor', 'y', ...
    'MarkerSize', 7);
else
    plot(X(versicolour, 1), X(versicolour, 2), 'kd', 'MarkerFaceColor', [0.5 0.5 0.5], ...
    'MarkerSize', 7);
end
if (a==3)||(a==0)
    plot(X(virginica, 1), X(virginica, 2), 'ko', 'MarkerFaceColor', 'g', ...
    'MarkerSize', 7);
else
    plot(X(virginica, 1), X(virginica, 2), 'ko', 'MarkerFaceColor', [0.5 0.5 0.5], ...
    'MarkerSize', 7);
end
ylabel('Sepal length in cm');               % Set the y-axis label
xlabel('petal length in cm');                % Set the X-axis label
legend('Setosa', 'Versicolour', 'Virginica', 'Location','northwest')


hold off;

end
