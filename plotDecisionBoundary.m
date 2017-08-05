function plotDecisionBoundary(w, X, y)
%Plot the data with the aditional decision boundary find whit the weight
%vector w.
m = size(X, 1);
num_labels = size(w,1);
X = [ones(m, 1) X];

figure('position', [100, 100, 1200, 300])
for k = 1:num_labels
    subplot(1,3,k)
    plotData(X(:,2:3), y, k);
    axis([2 10 0.5 7.5])
    hold on   
    
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./w(k,3)).*(w(k,2).*plot_x + w(k,1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y);
    title(sprintf('Decision Boundary %d',k));
    hold off
end

end
