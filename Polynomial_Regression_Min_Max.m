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

% Create a polynomial regression model using the training subset
X_train = TotalSteps(trainingIdx);
y_train = TotalDistance(trainingIdx);

degree = 2; % Degree of the polynomial
X_train_poly = [ones(length(X_train), 1), X_train, X_train.^degree];
coefficients_poly = (X_train_poly' * X_train_poly) \ (X_train_poly' * y_train);

% Test the model using the testing subset
X_test = TotalSteps(testingIdx);
y_test = TotalDistance(testingIdx);
X_test_poly = [ones(length(X_test), 1), X_test, X_test.^degree];
y_pred_poly = X_test_poly * coefficients_poly;

% Calculate and display the Mean Squared Error (MSE)
mse_poly = mean((y_test - y_pred_poly).^2);
fprintf('Polynomial Regression(Min-max scaling) Mean Squared Error: %.2f\n', mse_poly);

% Calculate and display the Mean Absolute Error (MAE)
mae_poly = mean(abs(y_test - y_pred_poly));
fprintf('Polynomial Regression(Min-max scaling) Mean Absolute Error: %.2f\n', mae_poly);

% Calculate and display the Pearson correlation coefficient (r value)
r_value = corr(y_test, y_pred_poly);
fprintf('Polynomial Regression(Min-max scaling) Pearson correlation coefficient (r value): %.2f\n', r_value);

% Calculate and display the Residual Standard Error (RSE)
n = length(y_test);
p = 2; % For a polynomial regression model of degree 2
RSS = sum((y_test - y_pred_poly).^2);
RSE = sqrt(RSS / (n - p - 1));
fprintf('Polynomial Regression(Min-max scaling) Residual Standard Error (RSE): %.2f\n', RSE);

% Predict TotalDistance for new TotalSteps
newTotalSteps = 10000; % Example value
newTotalSteps_scaled = (newTotalSteps - min(data.TotalSteps)) / (max(data.TotalSteps) - min(data.TotalSteps)); % Scale the new TotalSteps
X_new_poly = [1, newTotalSteps_scaled, newTotalSteps_scaled.^degree];
predictedTotalDistance = X_new_poly * coefficients_poly;

% Display the predicted TotalDistance
fprintf('Predicted TotalDistance for %d TotalSteps: %.2f\n', newTotalSteps, predictedTotalDistance);
