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

% Create a neural network regression model using the training subset
X_train = TotalSteps_scaled(trainingIdx)';
y_train = TotalDistance_scaled(trainingIdx)';

% Define the neural network architecture
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);

% Train the neural network regression model
[net,tr] = train(net,X_train,y_train);

% Test the neural network regression model using the testing subset
X_test = TotalSteps_scaled(testingIdx)';
y_test = TotalDistance_scaled(testingIdx)';
y_pred_nn = net(X_test);

% Calculate and display the Mean Squared Error (MSE)
mse_nn = mean((y_test - y_pred_nn).^2);
fprintf('Neural Network regression(Z-Score Scaling) Mean Squared Error: %.2f\n', mse_nn);

% Calculate and display the Mean Absolute Error (MAE)
mae_nn = mean(abs(y_test - y_pred_nn));
fprintf('Neural Network regression(Z-Score Scaling) Mean Absolute Error: %.2f\n', mae_nn);

% Calculate and display the Residual Standard Error (RSE)
n = length(y_test);
p = 1; % Number of predictor variables
RSS = sum((y_test - y_pred_nn).^2);
RSE = sqrt(RSS / (n - p - 1));
fprintf('Neural Network regression(Z-Score Scaling) Residual Standard Error (RSE): %.2f\n', RSE);

% Predict TotalDistance for new TotalSteps
newTotalSteps = 10000; % Example value
newTotalSteps_scaled = (newTotalSteps - mean(data.TotalSteps)) / std(data.TotalSteps); % Scale the new TotalSteps
predictedTotalDistance = net(newTotalSteps_scaled) * std(TotalDistance) + mean(TotalDistance);

% Display the predicted TotalDistance
fprintf('Predicted TotalDistance for %d TotalSteps: %.2f\n', newTotalSteps, predictedTotalDistance);
