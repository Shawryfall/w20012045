% Load dataset
data = readtable('FitBit data.csv');

% Extract data for TotalSteps and TotalDistance
TotalSteps = data.TotalSteps;
TotalDistance = data.TotalDistance;

% Scale TotalSteps using min-max scaling (normalization)
TotalSteps = (TotalSteps - min(TotalSteps)) / (max(TotalSteps) - min(TotalSteps));

% Split the dataset into training (80%) and testing (20%) subsets
rng('default'); % For reproducibility
splitRatio = 0.8;
splitIndex = floor(height(data) * splitRatio);
randomIndices = randperm(height(data));
trainingIdx = randomIndices(1:splitIndex);
testingIdx = randomIndices(splitIndex+1:end);

% Create a linear regression model using the training subset
X_train = TotalSteps(trainingIdx);
y_train = TotalDistance(trainingIdx);

% Perform linear regression
X_train = [ones(length(X_train), 1), X_train]; % Add a column of ones for the intercept term
coefficients = (X_train' * X_train) \ (X_train' * y_train); % Calculate coefficients using the normal equation

% Test the model using the testing subset
X_test = TotalSteps(testingIdx);
y_test = TotalDistance(testingIdx);
X_test = [ones(length(X_test), 1), X_test]; % Add a column of ones for the intercept term
y_pred = X_test * coefficients;

% Calculate and display the Mean Squared Error (MSE)
mse = mean((y_test - y_pred).^2);
fprintf('Linear regression(Min-Max Scaling) Mean Squared Error: %.2f\n', mse);

% Calculate and display the Mean Absolute Error (MAE)
mae = mean(abs(y_test - y_pred));
fprintf('Linear regression(Min-Max Scaling) Mean Absolute Error: %.2f\n', mae);

% Calculate and display the Pearson correlation coefficient (r value)
r = corr(y_test, y_pred);
fprintf('Linear regression(Min-Max Scaling) Pearson correlation coefficient (r value): %.2f\n', r);

% Calculate and display the Residual Standard Error (RSE)
n = length(y_test);
p = 1; % Number of predictor variables
RSS = sum((y_test - y_pred).^2);
RSE = sqrt(RSS / (n - p - 1));
fprintf('Linear regression(Min-Max Scaling) Residual Standard Error (RSE): %.2f\n', RSE);

% Predict TotalDistance for new TotalSteps
newTotalSteps = 10000; % Example value
newTotalSteps_scaled = (newTotalSteps - min(data.TotalSteps)) / (max(data.TotalSteps) - min(data.TotalSteps)); % Scale the new TotalSteps
X_new = [1, newTotalSteps_scaled]; % Add a column of ones for the intercept term
predictedTotalDistance = X_new * coefficients;

% Display the predicted TotalDistance
fprintf('Predicted TotalDistance for %d TotalSteps: %.2f\n', newTotalSteps, predictedTotalDistance);
