% Load dataset
data = readtable('FitBit data.csv');

% Extract data for TotalSteps and TotalDistance
TotalSteps = data.TotalSteps;
TotalDistance = data.TotalDistance;

% Scale TotalSteps and TotalDistance
TotalSteps_scaled = (TotalSteps - mean(TotalSteps)) / std(TotalSteps);
TotalDistance_scaled = (TotalDistance - mean(TotalDistance)) / std(TotalDistance);

% Split the dataset into training (80%) and testing (20%) subsets
rng('default'); % For reproducibility
splitRatio = 0.8;
splitIndex = floor(height(data) * splitRatio);
randomIndices = randperm(height(data));
trainingIdx = randomIndices(1:splitIndex);
testingIdx = randomIndices(splitIndex+1:end);

% Create a polynomial regression model using the training subset
X_train = TotalSteps_scaled(trainingIdx);
y_train = TotalDistance_scaled(trainingIdx);

% Train the polynomial regression model
mdl_pr = fitlm(X_train, y_train, 'poly2');

% Test the polynomial regression model using the testing subset
X_test = TotalSteps_scaled(testingIdx);
y_test = TotalDistance_scaled(testingIdx);
y_pred_pr = predict(mdl_pr, X_test);

% Calculate and display the Mean Squared Error (MSE)
mse_pr = mean((y_test - y_pred_pr).^2);
fprintf('Polynomial Regression(Z-score scaling) Mean Squared Error: %.2f\n', mse_pr);

% Calculate and display the Mean Absolute Error (MAE)
mae_pr = mean(abs(y_test - y_pred_pr));
fprintf('Polynomial Regression(Z-score scaling) Mean Absolute Error: %.2f\n', mae_pr);

% Calculate and display the Pearson correlation coefficient (r value)
r_value = corr(y_test, y_pred_pr);
fprintf('Polynomial Regression(Z-score scaling) Pearson correlation coefficient (r value): %.2f\n', r_value);

% Calculate and display the Residual Standard Error (RSE)
n = length(y_test);
p = 2; % For a polynomial regression model of degree 2
RSS = sum((y_test - y_pred_pr).^2);
RSE = sqrt(RSS / (n - p - 1));
fprintf('Polynomial Regression(Z-score scaling) Residual Standard Error (RSE): %.2f\n', RSE);

% Predict TotalDistance for new TotalSteps
newTotalSteps = 10000; % Example value
newTotalSteps_scaled = (newTotalSteps - mean(data.TotalSteps)) / std(data.TotalSteps); % Scale the new TotalSteps
predictedTotalDistance = predict(mdl_pr, newTotalSteps_scaled);

% Display the predicted TotalDistance
fprintf('Predicted TotalDistance for %d TotalSteps: %.2f\n', newTotalSteps, predictedTotalDistance * std(TotalDistance) + mean(TotalDistance));
