
%% Initialization
clear ; close all; clc
                          
                          %% Setup of parameters
samples=5000;               % number of samples to use (max 5000 for digits)
holdOut=1/3;                % parameter to split training data and test
num_labels = 10;            % 10 different labels for digits data set
lambda = 2;                 % Regularization parameter
iterations = 300;           % Number of iterations steepest gradient descent
iterations_QN = 25;         % Number for Quasi-Newton
alpha = 0.1;                % Steep size for gradient                                         
                          %% Optional parameters (y/n)
test_QNewton_opt='y';       % Test Quasi-Newton (conjugate gradient algorithm)(faster)
test_steepst_descent='n';   % Test gradian steepest descent (time consuming)
test_QNewton_descent='n';   % Test Quasi-Newton descent (time consuming)
test_optimal_lambda='n';    % Test different lambda parameters (time consuming)


%% =========== Part 1: Loading and Visualizing Data =============
% Load Data
fprintf('\nPart 1: Loading Data ...');
tic
load('digitDataSet.mat'); % training data stored in arrays X, y
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


%% =========== Part 3: Visualizing Data =============
fprintf('\nPart 3: Visualizing Data ...')
tic
rand_indices = randperm(n);
sel = X_train(rand_indices(1:100), :);
for i = 1:100
    R_i=sel(i,:);
    A(:,:,1,i)=reshape(R_i,20,20);
end

figure
thumbnails = A(:,:,:,1:100);
thumbnails = imresize(thumbnails, [64 64]);
montage(thumbnails)
title('Digit Data');

fprintf('(Done %f)',toc);


%% ============ Part 4: Training Logistic Regression Quasi-Newton (conjugate gradient algorithm) ============
fprintf('\nPart 4: Training Logistic Regression Quasi-Newton (conjugate gradient algorithm)...')
if test_QNewton_opt == 'y'
    % run gradient descent
    tic
    [all_w, j_h] = fmincgFunction(X_train, y_train, num_labels, lambda, iterations_QN);
    fprintf('(Done %f)',toc);
    % Plot the convergence graph
    plotGradient(j_h,'fmincg');
    fprintf('\nPredict values Quasi-Newton (conjugate gradient algorithm)...')
    tic
    pred = predict(all_w, X_train);
    pred_test = predict(all_w, X_test);
    fprintf('(Done %f)',toc);
    fprintf('\nTraining Set Accuracy: %f', mean(double(pred == y_train)) * 100);
    fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);
else
    fprintf('(Desactived)');
end


%% ============ Part 5: Training Logistic Regression Steepst Descent============
fprintf('\nPart 5: Training Logistic Regression steepestGradientDescent...')
if test_steepst_descent == 'y'
    % run gradient descent
    tic
    [all_w, j_h] = steepestGradientDescent(X_train, y_train, alpha, iterations, num_labels, lambda);
    fprintf('(Done %f)',toc);
    % Plot the convergence graph
    plotGradient(j_h,'Steepest Gradient Descent');
    fprintf('\nPredict values cross-validation steepestGradientDescent...')
    tic
    pred = predict(all_w, X_train);
    pred_test = predict(all_w, X_test);
    fprintf('(Done %f)',toc);
    fprintf('\nTraining Set Accuracy: %f', mean(double(pred == y_train)) * 100);
    fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);
else
    fprintf('(Desactived)');
end


%% ============ Part 6: Training Logistic Regression Quasi-Newton ============
fprintf('\nPart 6: Training Logistic Regression Quasi-Newton...')
if test_QNewton_descent == 'y'
    % run gradient descent
    tic
    [all_w] = quasiNewton(X_train, y_train, num_labels, lambda, iterations_QN);
    fprintf('(Done %f)',toc);
    fprintf('\nPredict values cross-validation Quasi-Newton...')
    tic
    pred = predict(all_w, X_train);
    pred_test = predict(all_w, X_test);
    fprintf('(Done %f)',toc);
    fprintf('\nTraining Set Accuracy: %f', mean(double(pred == y_train)) * 100);
    fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);
else
    fprintf('(Desactived)');
end

%% ================ Part 7: Tests Optimal Lambda ================

fprintf('\nPart 7: Optimazing parameters, testing differents lambda...')
if test_optimal_lambda == 'y'
    tic
    lambda_test = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
    plotBiasVsVariance(X_train, y_train, X_test, y_test, alpha, iterations_QN, num_labels, lambda_test, 3);
    fprintf('(Done %f)',toc);
else
    fprintf('(Desactived)');
end





