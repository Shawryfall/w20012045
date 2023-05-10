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

% Create a Random Forest Regression model using the training subset
X_train = TotalSteps(trainingIdx);
y_train = TotalDistance(trainingIdx);

% Train the Random Forest model
mdl_rf = fitrensemble(X_train, y_train, 'NumLearningCycles', 100, 'Method', 'Bag');

% Test the model using the testing subset
X_test = TotalSteps(testingIdx);
y_test = TotalDistance(testingIdx);
y_pred_rf = predict(mdl_rf, X_test);

% Calculate and display the Mean Squared Error (MSE)
mse_rf = mean((y_test - y_pred_rf).^2);
fprintf('Random Forest Regression(Min-max scaling) Mean Squared Error: %.2f\n', mse_rf);

% Calculate and display the Mean Absolute Error (MAE)
mae_rf = mean(abs(y_test - y_pred_rf));
fprintf('Random Forest Regression(Min-max scaling) Mean Absolute Error: %.2f\n', mae_rf);

% Calculate and display the Pearson correlation coefficient (r value)
r_value = corr(y_test, y_pred_rf);
fprintf('Random Forest Regression(Min-max scaling) Pearson correlation coefficient (r value): %.2f\n', r_value);

% Calculate and display the Residual Standard Error (RSE)
n = length(y_test);
p = length(mdl_rf.Trained{1}.PredictorNames); % Number of predictor variables
RSS = sum((y_test - y_pred_rf).^2);
RSE = sqrt(RSS / (n - p - 1));
fprintf('Random Forest Regression(Min-max scaling) Residual Standard Error (RSE): %.2f\n', RSE);

% Predict TotalDistance for new TotalSteps
newTotalSteps = 10000; % Example value
newTotalSteps_scaled = (newTotalSteps - min(data.TotalSteps)) / (max(data.TotalSteps) - min(data.TotalSteps)); % Scale the new TotalSteps
predictedTotalDistance = predict(mdl_rf, newTotalSteps_scaled);

% Display the predicted TotalDistance
fprintf('Predicted TotalDistance for %d TotalSteps: %.2f\n', newTotalSteps, predictedTotalDistance);
