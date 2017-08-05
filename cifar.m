
%% Initialization
clear ; close all; clc

                           %% Setup of parameters
samples=10000;              % number of samples to use
num_labels = 10;            % 10 labels, from 1 to 10
K = 95;                     % Number of principal components to use
lambda = 160;               % Regularization parameter
iterations = 1000;          % Number of iterations gradient descent
iterations_QN = 125;        % Number for Quasi-Newton
alpha = 0.01;               % Steep size for gradient
                           %% Optional parameters (y/n)
test_QNewton_opt='n';       % Test Quasi-Newton (conjugate gradient algorithm)(faster)
test_steepst_descent='n';   % Test gradian steepest descent (time consuming)
test_QNewton_descent='y';   % Test Quasi-Newton descent (time consuming)
test_optimal_lambda='n';    % Test different lambda parameters (time consuming)


%% =========== Part 0: Download the CIFAR-10 dataset=============
if ~exist('cifar-10-batches-mat','dir')
    cifar10Dataset = 'cifar-10-matlab';
    disp('Downloading 174MB CIFAR-10 dataset...');   
    websave([cifar10Dataset,'.tar.gz'],...
        ['https://www.cs.toronto.edu/~kriz/',cifar10Dataset,'.tar.gz']);
    gunzip([cifar10Dataset,'.tar.gz'])
    delete([cifar10Dataset,'.tar.gz'])
    untar([cifar10Dataset,'.tar'])
    delete([cifar10Dataset,'.tar'])
end 

%% =========== Part 1: Loading and Visualizing Data =============

fprintf('Loading and Visualizing Data ...')
tic

% Load only the first batch 10000 samples
load('cifar-10-batches-mat\data_batch_1.mat'); % training data stored in arrays X, y
X = data;
y= labels;

% ====================================================================================
% If you want to load 60000 samples uncomment this lines and comment the former
% remenber also change the number of samples on the setup parameter.

% load('cifar-10-batches-mat\data_batch_1.mat'); % training data stored in arrays X, y
% dataAux = data;
% labelsAux = labels;
% load('cifar-10-batches-mat\data_batch_2.mat'); % training data stored in arrays X, y
% dataAux = [dataAux;data];
% labelsAux = [labelsAux;labels];
% load('cifar-10-batches-mat\data_batch_3.mat'); % training data stored in arrays X, y
% dataAux = [dataAux;data];
% labelsAux = [labelsAux;labels];
% load('cifar-10-batches-mat\data_batch_4.mat'); % training data stored in arrays X, y
% dataAux = [dataAux;data];
% labelsAux = [labelsAux;labels];
% load('cifar-10-batches-mat\data_batch_5.mat'); % training data stored in arrays X, y
% dataAux = [dataAux;data];
% labelsAux = [labelsAux;labels];
% load('cifar-10-batches-mat\test_batch.mat'); % training data stored in arrays X, y
% X = [dataAux;data];
% y = [labelsAux;labels];
% ====================================================================================

m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(1:100, :);

sel = im2double(sel);

% separing the data selected into their different component images
R=sel(:,1:1024);
G=sel(:,1025:2048);
B=sel(:,2049:3072);

plotGrid(R,G,B,'Original');

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[R, mu, sigma] = featureNormalize(R);
[G, mu, sigma] = featureNormalize(G);
[B, mu, sigma] = featureNormalize(B);

plotGrid(R,G,B,'Normalized');

[UR, SR] = pca(R);
[UG, SG] = pca(G);
[UB, SB] = pca(B);

ZR = projectData(R, UR, K);
ZG = projectData(G, UG, K);
ZB = projectData(B, UB, K);
XR  = recoverData(ZR, UR, K);
XG  = recoverData(ZG, UG, K);
XB  = recoverData(ZB, UB, K);

plotGrid(XR,XG,XB,'Reduced');
fprintf('(Done %f)',toc);

fprintf('\nCalculing variance...');
tic
DR = diag(SR);
DG = diag(SG);
DB = diag(SB);
var = 1/3*(sum(DR(1:K))/sum(DR)+sum(DG(1:K))/sum(DG)+sum(DB(1:K))/sum(DB))*100;
fprintf('(Done %f)',toc);
fprintf('\nVariance ratained for K= %f is: %f percent \n', K, var);


%% =========== Part 2: Preparing all the Data =============

X = im2double(X);

% Separing all the data into their different component images
R=X(:,1:1024);
G=X(:,1025:2048);
B=X(:,2049:3072);

%  Before running PCA, it is important to first normalize X by subtracting 
%  the mean value from each feature
[R, mu, sigma] = featureNormalize(R);
[G, mu, sigma] = featureNormalize(G);
[B, mu, sigma] = featureNormalize(B);

% Appliying PCA to all the data
fprintf('\nAppliying PCA to all the data...');
tic
[UR, SR] = pca(R);
[UG, SG] = pca(G);
[UB, SB] = pca(B);

%projecting all the data with paramater K
ZR = projectData(R, UR, K);
ZG = projectData(G, UG, K);
ZB = projectData(B, UB, K);
X_reduced = [ZR,ZG,ZB];

fprintf('(Done %f)',toc);


%% =========== Part 3: Splitting the Data =============
fprintf('\nSplitting the data...');
tic
[train,test] = crossvalind('HoldOut',samples,0.1);

X_train = X_reduced(train,:);
y_train = y(train);

X_test = X_reduced(test,:);
y_test = y(test);

f = size(X_train, 2);
n = size(X_train, 1);
l = size(X_test, 1);
fprintf('(Done %f)',toc);
fprintf('\nNumber of features to use: %f',f);
fprintf('\nNumber of training: %f, and test samples: %f', n, l);


%% ============ Part 4: Trainning Logistic Regression Quasi-Newton (conjugate gradient algorithm) ============
fprintf('\nPart 4: Trainning Logistic Regression Quasi-Newton (conjugate gradient algorithm)...')
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


%% ============ Part 5: Trainning Logistic Regression Steepst Descent============
fprintf('\nPart 5: Trainning Logistic Regression steepestGradientDescent...')
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


%% ============ Part 6: Trainning Logistic Regression Quasi-Newton ============
fprintf('\nPart 6: Trainning Logistic Regression Quasi-Newton...')
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
    % lambda_test = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 110, 120, 130, 140, 150, 160, 170, 180];
    lambda_test = [0, 40, 80, 120, 160, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680];
    plotBiasVsVariance(X_train, y_train, X_test, y_test, alpha, iterations_QN, num_labels, lambda_test, 3);
    fprintf('(Done %f)',toc);
else
    fprintf('(Desactived)');
end
