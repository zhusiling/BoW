% Initialization
clear ; clc; close all;

% Location of the compressed data set
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% Store the output in a temporary folder
outputFolder = fullfile(tempdir, 'caltech101'); % define output folder

if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end

rootFolder = fullfile(outputFolder, '101_ObjectCategories');

categories = {'airplanes', 'ferry', 'laptop'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

tbl = countEachLabel(imds)

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

[trainingSet, validationSet] = splitEachLabel(imds, 0.3, 'randomize');

% Find the first instance of an image for each category
airplanes = find(trainingSet.Labels == 'airplanes', 1);
ferry = find(trainingSet.Labels == 'ferry', 1);
laptop = find(trainingSet.Labels == 'laptop', 1);

% figure

subplot(1,3,1);
imshow(readimage(trainingSet,airplanes))
subplot(1,3,2);
imshow(readimage(trainingSet,ferry))
subplot(1,3,3);
imshow(readimage(trainingSet,laptop))

bag = bagOfFeatures(imageSet(trainingSet.Files));


img = readimage(imds, 1);
featureVector = encode(bag, img);

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('Visual word occurrences')
xlabel('Visual word index')
ylabel('Frequency of occurrence')

% build my new feature vector with values of histogram of each image
% also conver the 
X_train = zeros(60,500);
for i = 1:length (trainingSet.Files)
    img = readimage(trainingSet, i);
    X_train(i,:) = encode(bag, img);
    if (trainingSet.Labels(i) == 'airplanes')
        y_train(i)=1;
    elseif (trainingSet.Labels(i) == 'ferry')
        y_train(i)=2;
    elseif (trainingSet.Labels(i) == 'laptop')
        y_train(i)=3;
    end
end

fprintf('\n\nPart 4: Training Logistic Regression steepestGradientDescent...')
tic
% run gradient descent
[all_w, j_h] = steepestGradientDescent(X_train, y_train', 0.1, 400, 3, 1);
fprintf('(Done %f)',toc);

% Plot the convergence graph
plotGradient(j_h,'Steepest Gradient Descent');

pred = predict(all_w, X_train);
fprintf('\nTraining Set Accuracy: %f', mean(double(pred == y_train')) * 100);


% Encode the test data to evaluate it 
X_test = zeros(141,500);
for i = 1:length (validationSet.Files)
    img = readimage(validationSet, i);
    X_test(i,:) = encode(bag, img);
    if (validationSet.Labels(i) == 'airplanes')
        y_test(i)=1;
    elseif (validationSet.Labels(i) == 'ferry')
        y_test(i)=2;
    elseif (validationSet.Labels(i) == 'laptop')
        y_test(i)=3;
    end
end

pred_test = predict(all_w, X_test);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred_test == y_test')) * 100);
