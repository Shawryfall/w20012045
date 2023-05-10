% Load dataset
data = readtable('FitBit data.csv');

% Extract data for TotalSteps and TotalDistance
TotalSteps = data.TotalSteps;
TotalDistance = data.TotalDistance;

% Scale TotalSteps
TotalSteps = (TotalSteps - min(TotalSteps)) / (max(TotalSteps) - min(TotalSteps));

% Split the dataset into training (80%) and testing (20%) subsets
rng('default'); % For reproducibility
splitRatio = 0.8;
splitIndex = floor(height(data) * splitRatio);
randomIndices = randperm(height(data));
trainingIdx = randomIndices(1:splitIndex);
testingIdx = randomIndices(splitIndex+1:end);

% Create a Support Vector Regression model using the training subset
X_train = TotalSteps(trainingIdx);
y_train = TotalDistance(trainingIdx);

% Train the SVR model
mdl_svr = fitrsvm(X_train, y_train, 'KernelFunction', 'rbf', 'Standardize', true);

% Test the model using the testing subset
X_test = TotalSteps(testingIdx);
y_test = TotalDistance(testingIdx);
y_pred_svr = predict(mdl_svr, X_test);

% Calculate and display the Mean Squared Error (MSE)
mse_svr = mean((y_test - y_pred_svr).^2);
fprintf('Support Vector Regression(Min-max scaling) Mean Squared Error: %.2f\n', mse_svr);

% Calculate and display the Mean Absolute Error (MAE)
mae_svr = mean(abs(y_test - y_pred_svr));
fprintf('Support Vector Regression(Min-max scaling) Mean Absolute Error: %.2f\n', mae_svr);

% Calculate and display the Pearson correlation coefficient (r value)
r_value = corr(y_test, y_pred_svr);
fprintf('Support Vector Regression(Min-max scaling) Pearson correlation coefficient (r value): %.2f\n', r_value);

% Calculate and display the Residual Standard Error (RSE)
n = length(y_test);
p = length(mdl_svr.PredictorNames); % Number of predictor variables
RSS = sum((y_test - y_pred_svr).^2);
RSE = sqrt(RSS / (n - p - 1));
fprintf('Support Vector Regression(Min-max scaling) Residual Standard Error (RSE): %.2f\n', RSE);

% Predict TotalDistance for new TotalSteps
newTotalSteps = 10000; % Example value
newTotalSteps_scaled = (newTotalSteps - min(data.TotalSteps)) / (max(data.TotalSteps) - min(data.TotalSteps)); % Scale the new TotalSteps
predictedTotalDistance = predict(mdl_svr, newTotalSteps_scaled);

% Display the predicted TotalDistance
fprintf('Predicted TotalDistance for %d TotalSteps: %.2f\n', newTotalSteps, predictedTotalDistance);
