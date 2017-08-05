

%% Initialization
clear ; clc; close all;

%% Setup of parameters
samples=150;                % number of samples to use (max 150 for iris)
holdOut=1/3;                % parameter to split training data and test
num_labels = 3;             % 3 labels for iris data set
lambda = 0;                 % Regularization parameter
iterations = 600;           % Number of iterations gradient descent
alpha = 0.1;                % Steep size for gradient

%% =========== Part 1: Loading and Visualizing Data =============

% Load Data
fprintf('Part 1: Loading and Visualizing Data ...')
tic
load('irisDataSetSimple.mat'); % training data stored in arrays X, y

plotData(X, y, 0);
fprintf('(Done %f)',toc);

%% =========== Part 2: Splitting Data =============
fprintf('\nPart 2: Splitting the data...');
tic
[train,test] = crossvalind('HoldOut',samples,holdOut);

X_train = X(train,:);
y_train = y(train);

X_test = X(test,:);
y_test = y(test);

f = size(X_train, 2);
n = size(X_train, 1);
l = size(X_test, 1);

fprintf('(Done %f)',toc);
fprintf('\nNumber of features to use: %f',f);
fprintf('\nNumber of training: %f, and test samples: %f', n, l);


%% ============ Part 3: Optimazing parameters ============
fprintf('\nPart 3: Optimazing parameters, testing differents lambda...')
tic
lambda_test = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1];
plotBiasVsVariance(X_train, y_train, X_test, y_test, alpha, iterations, num_labels, lambda_test, 1);

fprintf('(Done %f)',toc);


%% ============ Part 4: Training Logistic Regression Steepst Descent============

fprintf('\nPart 4: Training Logistic Regression steepestGradientDescent...')
tic
% run gradient descent
[all_w, j_h] = steepestGradientDescent(X_train, y_train, alpha, iterations, num_labels, lambda);
fprintf('(Done %f)',toc);

% Plot the convergence graph
plotGradient(j_h,'Steepest Gradient');
plotDecisionBoundary(all_w, X_train, y_train);

fprintf('\nChecking gradian...\n')
checkgrad('regCostFunction', all_w(1,:)', 1e-6, X_train, y_train, lambda)


%% ================ Part 5: Predict values cross-validation ================
fprintf('\nPart 5: Predict values cross-validation...')
tic
pred = predict(all_w, X_train);
pred_test = predict(all_w, X_test);
fprintf('(Done %f)',toc);
fprintf('\nTraining Set Accuracy: %f', mean(double(pred == y_train)) * 100);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);


